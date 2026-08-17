import os
import re
from pathlib import Path

from dotenv import load_dotenv

try:
    import ftfy
except ImportError:
    ftfy = None

# The chunker reads COHERE_API_KEY straight from the environment, so the .env
# has to be loaded before then. Anchored to the repo root rather than the
# working directory so `python -m src.indexing.parse` works from anywhere.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_semantic_chunker = None


def get_semantic_chunker():
    """Lazily initialize the semantic chunker using Cohere embeddings."""
    global _semantic_chunker
    if _semantic_chunker is None:
        from langchain_cohere import CohereEmbeddings
        from langchain_experimental.text_splitter import SemanticChunker

        api_key = os.environ.get("COHERE_API_KEY")
        embeddings = CohereEmbeddings(
            cohere_api_key=api_key,
            model="embed-v4.0",
            user_agent="semantic-chunking"
        )
        _semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95.0,  # Adjust lower (e.g. 80-90) for smaller chunks
        )
    return _semantic_chunker


def clean_text_for_embedding(text: str) -> str:
    """Sanitize raw text extracted from any PDF without publisher-specific rules."""
    if not text:
        return ""

    # 1. Fix encoding and unicode corruptions if ftfy is available
    if ftfy is not None:
        text = ftfy.fix_text(text)

    # 2. Remove HTML / XML tags (e.g. <sup>1,28</sup>)
    text = re.sub(r"<[^>]+>", "", text)

    # 3. Strip standalone URLs and emails
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # 4. Remove bracketed numeric citations (e.g., "[1, 28]" or "[1-3]")
    text = re.sub(r"\[\d+(?:[,–-]\s*\d+)*\]", "", text)

    # 5. Fix Markdown/Math symbol corruptions (e.g., "_ P _ < .001" -> "P < .001")
    text = re.sub(r"_\s*([A-Za-z])\s*_", r"\1", text)

    # 6. Collapse all multi-whitespace and linebreaks into single spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Clean up spaces before punctuation left by stripped citations
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


def chunk_text_semantically(text: str) -> list[str]:
    """Clean text and chunk it based on semantic transitions in Cohere embed-v4.0 vector space."""
    cleaned = clean_text_for_embedding(text)
    if not cleaned:
        return []

    chunker = get_semantic_chunker()
    docs = chunker.create_documents([cleaned])
    return [doc.page_content for doc in docs]


