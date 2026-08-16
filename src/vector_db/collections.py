"""
Load the embedded corpus into Chroma and search it.

Vectors are computed ahead of time by src.indexing.embeddings, so they are
passed to Chroma directly and no embedding function is attached to the
collection. Queries are embedded on demand with input_type="search_query".

Usage:  python -m src.vector_db.collections          # build the index
        python -m src.vector_db.collections "query"  # build, then search
"""

import json
import os
import sys

import numpy as np

from src.indexing.embeddings import embed_query, load_chunks
from src.vector_db.client import get_client, reset
from src.vector_db.schemas import (
    COLLECTION_NAME,
    DISTANCE_METRIC,
    build_where,
    to_metadata,
)

VECTORS_PATH = os.path.join("data", "embeddings", "vectors.npy")
IDS_PATH = os.path.join("data", "embeddings", "ids.json")
ADD_BATCH = 500


def get_collection():
    """Fetch the collection, creating it empty if it does not exist yet."""
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": DISTANCE_METRIC},
    )


def build():
    """Rebuild the collection from chunks.jsonl + vectors.npy."""
    chunks = load_chunks()
    vectors = np.load(VECTORS_PATH)
    with open(IDS_PATH, encoding="utf-8") as fh:
        ids = json.load(fh)

    if not (len(chunks) == len(vectors) == len(ids)):
        raise SystemExit(
            f"length mismatch: {len(chunks)} chunks, {len(vectors)} vectors, "
            f"{len(ids)} ids - rerun src.indexing.embeddings"
        )
    if [c["id"] for c in chunks] != ids:
        raise SystemExit("ids.json is out of order with chunks.jsonl - reembed")

    reset()  # a rebuild should not leave stale rows behind
    collection = get_collection()

    for start in range(0, len(chunks), ADD_BATCH):
        stop = min(start + ADD_BATCH, len(chunks))
        collection.add(
            ids=ids[start:stop],
            embeddings=vectors[start:stop].tolist(),
            documents=[c["text"] for c in chunks[start:stop]],
            metadatas=[to_metadata(c) for c in chunks[start:stop]],
        )
        print(f"  indexed {stop}/{len(chunks)}")

    print(f"\ncollection '{COLLECTION_NAME}': {collection.count()} vectors")
    return collection


def search(query, k=5, corpus=None, topic=None, section=None):
    """
    Semantic search with optional metadata filtering.

    Returns dicts with a `score` in [0, 1] where higher is closer. Chroma
    reports cosine *distance*, so it is converted here to keep the sign
    intuitive for callers.
    """
    collection = get_collection()
    result = collection.query(
        query_embeddings=[embed_query(query).tolist()],
        n_results=k,
        where=build_where(corpus, topic, section),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for chunk_id, doc, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append({"id": chunk_id, "text": doc, "score": 1.0 - dist, **meta})
    return hits


def main():
    build()
    query = sys.argv[1] if len(sys.argv) > 1 else "Who should be screened for hepatitis C?"
    print(f"\nquery: {query}")
    for hit in search(query, k=3):
        print(f"  {hit['score']:.3f} [{hit['corpus']}] {hit['topic'][:34]} > {hit['heading'][:40]}")


if __name__ == "__main__":
    main()
