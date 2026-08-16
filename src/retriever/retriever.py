import os
import numpy as np
from src.config import CHUNKS_PATH, VECTORS_PATH
from src.indexing.embeddings import embed_query, load_chunks

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve top-k most relevant chunks for a given user query."""
    if not os.path.exists(VECTORS_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "Index files missing! Run with --index first to build the search database."
        )

    # Load pre-indexed data
    chunks = load_chunks(CHUNKS_PATH)
    vectors = np.load(VECTORS_PATH)

    # Embed query vector
    query_vec = embed_query(query)

    # Compute cosine similarity
    similarities = np.dot(vectors, query_vec)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["score"] = float(similarities[idx])
        results.append(chunk)

    return results
