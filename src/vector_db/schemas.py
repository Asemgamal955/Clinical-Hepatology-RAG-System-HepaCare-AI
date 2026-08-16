"""
What a chunk looks like inside the vector store.

Chroma metadata values must be scalars (str/int/float/bool), so this is a flat
projection of a chunk record. Only fields listed here are filterable at query
time - the chunk text itself lives in Chroma's `documents`, not in metadata.
"""

COLLECTION_NAME = "liver_rag"
DIMENSION = 1024  # must match src.indexing.embeddings.DIMENSION

# Fields promoted to filterable metadata.
METADATA_FIELDS = ("corpus", "topic", "section", "heading", "url", "source")

# Cosine, because the Cohere vectors are unit-normalized.
DISTANCE_METRIC = "cosine"


def to_metadata(chunk):
    """Flatten a chunk record into Chroma metadata."""
    return {field: str(chunk.get(field, "")) for field in METADATA_FIELDS}


def build_where(corpus=None, topic=None, section=None):
    """
    Compose a Chroma `where` filter. Returns None when nothing is constrained,
    which Chroma treats as an unfiltered search.

    Chroma needs an explicit $and once there is more than one condition.
    """
    clauses = []
    if corpus:
        clauses.append({"corpus": corpus})
    if topic:
        clauses.append({"topic": topic})
    if section:
        clauses.append({"section": section})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
