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
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from src.config import LLM_MODEL

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

_PROMPT = """You rewrite patient questions about liver disease for a search engine.

Return ONLY a JSON object with exactly these two keys:
  "dense_query":  a clear clinical rephrasing as a full sentence or noun phrase,
                  using standard medical terminology.
  "sparse_query": the key medical terms only, space separated, no stopwords,
                  no punctuation.

Keep every clinical term, acronym, and lab name that appears in the question.
Do not invent conditions the user did not mention.

Question: {query}"""

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
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key or key == "your-key-here":
            return None
        from google import genai

        _client = genai.Client(api_key=key)
    return _client


def _llm_rewrite(query):
    """Ask the LLM for both forms. Returns None if unavailable or malformed."""
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=_PROMPT.format(query=query),
            config={"response_mime_type": "application/json"},
        )
        data = json.loads(response.text)
        dense = str(data.get("dense_query", "")).strip()
        sparse = str(data.get("sparse_query", "")).strip()
        if dense and sparse:
            return dense, sparse
    except Exception as exc:
        print(f"  query rewrite unavailable ({type(exc).__name__}), using dictionary only")
    return None


def parse_query(query, use_llm=True):
    """
    Build a ParsedQuery. Never raises and never returns empty fields - if every
    optional stage fails the caller still gets the original question back.
    """
    query = (query or "").strip()
    if not query:
        return ParsedQuery(raw=query, dense_query=query, sparse_query=query)

    expanded, applied = expand_clinical_terms(query)

    rewritten = _llm_rewrite(expanded) if use_llm else None
    if rewritten:
        dense, sparse = rewritten
        return ParsedQuery(query, dense, sparse, applied, used_llm=True)

    return ParsedQuery(query, expanded, keywords_only(expanded), applied, used_llm=False)


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
