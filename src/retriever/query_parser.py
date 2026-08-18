"""
Query understanding: turn what the user typed into what each retrieval leg wants.

Two stages, deliberately in this order:

1. A clinical synonym dictionary. Deterministic, free, offline. Patients write
   "fatty liver"; the corpus writes "NAFLD". Every target term here was checked
   to actually occur in chunks.jsonl - mapping to a word the corpus never uses
   would be worse than not expanding at all.

2. An LLM rewrite producing two forms of the question, because the legs want
   different things. Dense retrieval reads meaning, so it wants a fluent
   clinical sentence. BM25 counts token overlap, so "what should i eat if my"
   is pure noise to it and it wants keywords only.

The LLM stage is optional. Without GEMINI_API_KEY the parser still expands
terms and strips stopwords, so retrieval never depends on the network.

No metadata filters are produced. A wrong corpus/topic guess silently removes
good chunks - and on a 348-chunk corpus there is nothing to gain by narrowing
the search in the first place. Filters stay available on hybrid_search() for
callers who genuinely know the constraint.

Usage:  python -m src.retriever.query_parser "your question"
"""

import json
import os
import re
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path
from google import genai
from google.genai import types  # Import type
from dotenv import load_dotenv

from src.config import LLM_MODEL

MAX_RETRIES = 4

# Fail loudly instead of degrading to dictionary-only. Benchmarks set this so
# a rate-limited run cannot be mistaken for a real measurement.
STRICT = os.environ.get("STRICT_QUERY_REWRITE") == "1"

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Colloquial or abbreviated phrasing -> the term the corpus actually uses.
# Expansions are appended rather than substituted, so the user's own wording
# still matches documents that happen to use it.
CLINICAL_TERMS = {
    "fatty liver": "NAFLD nonalcoholic fatty liver disease",
    "liver fat": "NAFLD",
    "hep a": "hepatitis A",
    "hep b": "hepatitis B",
    "hep c": "hepatitis C",
    "hep d": "hepatitis D",
    "hep e": "hepatitis E",
    "hbv": "hepatitis B virus",
    "hcv": "hepatitis C virus",
    "hbsag": "hepatitis B surface antigen",
    "pbc": "primary biliary cholangitis",
    "psc": "primary sclerosing cholangitis",
    "yellow skin": "jaundice",
    "yellow eyes": "jaundice",
    "yellowing": "jaundice",
    "liver scarring": "cirrhosis",
    "scarred liver": "cirrhosis",
    "iron overload": "hemochromatosis",
    "too much iron": "hemochromatosis",
    "copper buildup": "Wilson disease",
    "too much copper": "Wilson disease",
    "fluid in belly": "ascites",
    "swollen belly": "ascites",
    "belly swelling": "ascites",
    "confusion from liver": "hepatic encephalopathy",
    "new liver": "liver transplant",
    "liver operation": "liver transplant",
    "screening guideline": "USPSTF recommendation",
    "task force": "USPSTF",
}

# Dropped from the BM25 form only. BM25 scores by token overlap, so question
# scaffolding actively competes with the terms that matter.
STOPWORDS = {
    "a", "about", "am", "an", "and", "any", "are", "as", "at", "be", "been",
    "by", "can", "could", "do", "does", "for", "from", "get", "getting", "give",
    "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its",
    "long", "many", "me", "much", "my", "need", "of", "on", "or", "our", "should",
    "so", "some", "tell", "that", "the", "their", "them", "there", "these",
    "they", "this", "to", "was", "we", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
}

_CLINICAL_DICT_TEXT = "\n".join(
    f"  - '{k}' -> '{v}'" for k, v in CLINICAL_TERMS.items()
)

_PROMPT = f"""You rewrite patient questions about liver disease for a search engine (Dense Vector + BM25 Sparse).

### Canonical Term Mappings (Preferred Corpus Terms):
When colloquial or abbreviated terms match these keys, prefer these target terms:
{_CLINICAL_DICT_TEXT}

### Instructions:
Return ONLY a valid JSON object with exactly two keys:
1. "dense_query": A clear clinical rephrasing as a full sentence using standard medical terminology.
2. "sparse_query": Key medical terms only, space-separated, no stopwords, no punctuation.

### Rules:
- The dictionary above provides preferred terms, but you are NOT strictly restricted to it.
- If the query mentions symptoms, conditions, or labs NOT listed in the dictionary, use your clinical knowledge to select and include the appropriate medical terms and keywords.
- Include the canonical dictionary mappings whenever a matching colloquial phrase is present.
- Retain core user keywords while removing non-informative stopwords/question scaffolding.

Question: {{query}}
JSON Output:"""

_client = None


@dataclass
class ParsedQuery:
    """What the retrieval legs consume. `raw` is kept for logging and display."""

    raw: str
    dense_query: str
    sparse_query: str
    expansions: list = field(default_factory=list)
    used_llm: bool = False


def expand_clinical_terms(query):
    """
    Append the canonical term for any colloquial phrase found.

    Matching is on token presence rather than exact substring, because people
    do not type phrases in dictionary order - "my liver is fatty" and "eyes are
    yellow" both have to hit, and a substring match catches neither. Phrases
    here are two or three specific words, so co-occurrence is a strong enough
    signal.

    Appending rather than replacing is deliberate: substitution can destroy a
    query when a phrase matches inside a larger one, and keeping both forms
    costs nothing in a bag-of-words leg.
    """
    query_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)}
    expanded, applied = query, []

    for phrase, canonical in CLINICAL_TERMS.items():
        phrase_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", phrase)}
        if not phrase_tokens <= query_tokens:
            continue
        # Skip if the canonical term is already present anyway.
        if re.search(rf"\b{re.escape(canonical.split()[0])}\b", expanded, re.IGNORECASE):
            continue
        expanded = f"{expanded} {canonical}"
        applied.append(f"{phrase} -> {canonical}")
    return expanded, applied


def keywords_only(text):
    """Strip stopwords and punctuation. The deterministic BM25 form."""
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    kept = [t for t in tokens if t.lower() not in STOPWORDS]
    return " ".join(kept) if kept else text


def _get_client():
    """
    Prefer Lightning, fall back to the Google API.

    Google's free tier allows 15 requests/minute, which aborted benchmark runs
    part-way through. Lightning serves the same model without that cap, so it
    is used whenever LIGHTNING_API_KEY is present.

    Returns (client, kind) where kind is "lightning" or "gemini".
    """
    global _client
    if _client is None:
        lightning_key = os.environ.get("LIGHTNING_API_KEY")
        if lightning_key and lightning_key != "your-key-here":
            from src.generation.lightning_llm import LightningLLM

            _client = (LightningLLM(api_key=lightning_key), "lightning")
        else:
            key = os.environ.get("GEMINI_API_KEY")
            if not key or key == "your-key-here":
                return None
            from google import genai

            _client = (genai.Client(api_key=key), "gemini")
    return _client


def _retry_delay(exc, attempt):
    """Seconds to wait. Google reports its own retryDelay on a 429; honour it."""
    match = re.search(r"'retryDelay':\s*'(\d+)s'", str(exc))
    if match:
        return int(match.group(1)) + 1
    return min(2 ** attempt, 60)


def _llm_rewrite(query):
    """
    Ask the LLM for both query forms.

    On the free tier Gemini allows 15 requests/minute and returns 429 beyond
    that. Silently dropping to the dictionary was wrong for evaluation: the
    parse-enabled configs got scored without the rewrite they exist to test,
    so their numbers looked worse than the feature actually is. Rate limits
    are now waited out rather than swallowed.

    STRICT_QUERY_REWRITE=1 turns any remaining failure into an exception, so
    a benchmark aborts instead of quietly reporting a degraded run.
    """
    resolved = _get_client()
    if resolved is None:
        if STRICT:
            raise RuntimeError("no LLM key set and STRICT_QUERY_REWRITE=1")
        return None
    client, kind = resolved

    last = None
    for attempt in range(MAX_RETRIES):
        try:
            if kind == "lightning":
                raw = client.chat(_PROMPT.format(query=query),
                                  temperature=0.0, json_mode=True)
            else:
                raw = client.models.generate_content(
                    model=LLM_MODEL,
                    contents=_PROMPT.format(query=query),
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                ).text
            data = json.loads(raw)
            dense = str(data.get("dense_query", "")).strip()
            sparse = str(data.get("sparse_query", "")).strip()
            if dense and sparse:
                return dense, sparse
            last = ValueError("LLM returned empty dense_query/sparse_query")
        except Exception as exc:
            last = exc
            if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                wait = _retry_delay(exc, attempt)
                if attempt < MAX_RETRIES - 1:
                    print(f"  rate limited, waiting {wait}s ({attempt + 1}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
            elif attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
        break

    if STRICT:
        raise RuntimeError(f"query rewrite failed after {MAX_RETRIES} attempts: {last}")
    print(f"  query rewrite unavailable ({type(last).__name__}), using dictionary only")
    return None


_cache = {}


def parse_query(query, use_llm=True):
    """
    Build a ParsedQuery. Never raises and never returns empty fields - if every
    optional stage fails the caller still gets the original question back.

    Results are memoised per process. The rewrite is deterministic for a given
    question, and benchmarks sweep the same query set across several k values,
    so without this a 35-query set costs 140 LLM calls instead of 35 - enough
    to spend most of a run sitting out rate limits.
    """
    query = (query or "").strip()
    if not query:
        return ParsedQuery(raw=query, dense_query=query, sparse_query=query)

    key = (query, use_llm)
    if key in _cache:
        return _cache[key]

    expanded, applied = expand_clinical_terms(query)

    rewritten = _llm_rewrite(expanded) if use_llm else None
    if rewritten:
        dense, sparse = rewritten
        parsed = ParsedQuery(query, dense, sparse, applied, used_llm=True)
    else:
        parsed = ParsedQuery(query, expanded, keywords_only(expanded), applied, used_llm=False)

    _cache[key] = parsed
    return parsed


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "what should i eat if my liver is fatty"
    parsed = parse_query(query)
    print(f"raw    : {parsed.raw}")
    print(f"dense  : {parsed.dense_query}")
    print(f"sparse : {parsed.sparse_query}")
    print(f"expands: {parsed.expansions or 'none'}")
    print(f"llm    : {parsed.used_llm}")


if __name__ == "__main__":
    main()
