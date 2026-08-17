"""
Dense retrieval.

A thin wrapper over the Chroma collection so callers depend on a retriever
interface rather than on the vector store directly - the hybrid retriever and
the generation pipeline both go through here.

Usage:  python -m src.retriever.retriever "your question"
"""

import sys

from src.vector_db.collections import search as _chroma_search


def dense_search(query, k=5, corpus=None, topic=None, section=None):
    """
    Semantic search. Returns hits ordered by cosine similarity, each carrying
    `score` (higher is closer), `text`, and the chunk metadata.
    """
    hits = _chroma_search(query, k=k, corpus=corpus, topic=topic, section=section)
    for rank, hit in enumerate(hits, 1):
        hit["dense_rank"] = rank
        hit["dense_score"] = hit["score"]
    return hits


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "What causes cirrhosis?"
    print(f"query: {query}\n")
    for hit in dense_search(query, k=5):
        print(f"  {hit['score']:.3f} [{hit['corpus']}] {hit['heading'][:52]}")


if __name__ == "__main__":
    main()
