"""
Query term expansion utilities for BM25 retrieval.

Provides:
  - CLINICAL_TERMS  : colloquial/abbreviated phrase -> corpus canonical term mapping
  - STOPWORDS       : tokens to strip before BM25 scoring
  - expand_clinical_terms(query)  -> (expanded_str, applied_list)
  - keywords_only(text)           -> stopword-stripped keyword string for BM25
"""

import re

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


def expand_clinical_terms(query: str) -> tuple[str, list[str]]:
    """
    Append the canonical term for any colloquial phrase found in query.

    Returns (expanded_query, list_of_applied_mappings).
    Matching is token-based so word order does not matter.
    """
    query_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)}
    expanded, applied = query, []

    for phrase, canonical in CLINICAL_TERMS.items():
        phrase_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", phrase)}
        if not phrase_tokens <= query_tokens:
            continue
        # Skip if the canonical term is already present.
        if re.search(rf"\b{re.escape(canonical.split()[0])}\b", expanded, re.IGNORECASE):
            continue
        expanded = f"{expanded} {canonical}"
        applied.append(f"{phrase} -> {canonical}")

    return expanded, applied


def keywords_only(text: str) -> str:
    """Strip stopwords and punctuation — the deterministic BM25 query form."""
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    kept = [t for t in tokens if t.lower() not in STOPWORDS]
    return " ".join(kept) if kept else text


def bm25_query(query: str) -> str:
    """
    Convenience: expand clinical terms then strip stopwords.
    Returns the keyword string to feed directly into BM25.
    """
    expanded, _ = expand_clinical_terms(query)
    return keywords_only(expanded)
