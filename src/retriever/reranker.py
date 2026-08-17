"""
Cross-encoder reranking with Cohere rerank-v3.5.

Retrieval scores a query and a chunk separately and compares the two vectors,
which is fast enough to run over the whole corpus but throws away any
word-level interaction between them. A reranker reads the query and the chunk
together and scores the pair directly. That is far more accurate and far more
expensive, so it only ever sees the handful of candidates retrieval already
shortlisted.

Usage:  python -m src.retriever.reranker "your question"
"""

import os
import sys
import time

import cohere
from dotenv import load_dotenv

MODEL = "rerank-v3.5"
MAX_RETRIES = 4

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("COHERE_API_KEY")
        if not key:
            raise SystemExit("COHERE_API_KEY not set - add it to .env")
        _client = cohere.ClientV2(key)
    return _client


def rerank(query, hits, top_n=5, client=None):
    """
    Reorder `hits` by relevance to `query`.

    Each returned hit gets `rerank_score` (0-1) and keeps `fused_score` /
    `dense_rank` / `bm25_rank` so the retrieval path stays inspectable. `score`
    is overwritten with the rerank score, since that is now the ranking signal.

    Returns hits unchanged if the API fails - a degraded ordering beats a dead
    demo.
    """
    if not hits:
        return []

    client = client or get_client()
    documents = [hit["text"] for hit in hits]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.rerank(
                model=MODEL,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
            )
            break
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                print(f"  rerank failed ({type(exc).__name__}), keeping retrieval order")
                return hits[:top_n]
            time.sleep(2**attempt)

    ranked = []
    for position, result in enumerate(response.results, 1):
        hit = dict(hits[result.index])
        hit["rerank_score"] = float(result.relevance_score)
        hit["rerank_position"] = position
        hit["retrieval_rank"] = result.index + 1
        hit["score"] = hit["rerank_score"]
        ranked.append(hit)
    return ranked


def main():
    from src.retriever.hybrid import hybrid_search

    query = sys.argv[1] if len(sys.argv) > 1 else "What causes cirrhosis?"
    print(f"query: {query}\n")
    for hit in hybrid_search(query, k=5, rerank=True):
        moved = hit["retrieval_rank"] - hit["rerank_position"]
        arrow = f"{moved:+d}" if moved else " ="
        print(f"  {hit['score']:.3f} [{arrow}] {hit['heading'][:52]}")


if __name__ == "__main__":
    main()
