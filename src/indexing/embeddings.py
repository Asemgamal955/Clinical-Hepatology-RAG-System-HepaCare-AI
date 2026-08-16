"""
Embed the chunked corpus with Cohere embed-v4.0.

Documents and queries are embedded with different input_type values, which is
what the model expects for retrieval - a query and the passage answering it are
not the same kind of text, and telling the model which is which improves the
match.

Input:   data/processed/chunks.jsonl
Outputs: data/embeddings/vectors.npy   float32 [n_chunks, DIMENSION]
         data/embeddings/ids.json      chunk ids, aligned row-for-row

Usage:  python -m src.indexing.embeddings
"""

import json
import os
import time
from pathlib import Path

import cohere
import numpy as np
from dotenv import load_dotenv

CHUNKS_PATH = os.path.join("data", "processed", "chunks.jsonl")
OUT_DIR = os.path.join("data", "embeddings")

MODEL = "embed-v4.0"
DIMENSION = 1024  # v4 also supports 256/512/1536
BATCH_SIZE = 96  # Cohere's per-call maximum
MAX_RETRIES = 5

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def get_client():
    key = os.environ.get("COHERE_API_KEY")
    if not key:
        raise SystemExit("COHERE_API_KEY not set - add it to .env")
    return cohere.ClientV2(key)


def _embed(client, texts, input_type):
    """One call, retrying with backoff so a rate limit doesn't kill the run."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embed(
                texts=texts,
                model=MODEL,
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=DIMENSION,
            )
            return response.embeddings.float_
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2**attempt
            print(f"  retry {attempt + 1}/{MAX_RETRIES} in {wait}s ({type(exc).__name__})")
            time.sleep(wait)


def embed_documents(texts, client=None):
    """Embed chunks for indexing."""
    client = client or get_client()
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vectors.extend(_embed(client, batch, "search_document"))
        print(f"  embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text, client=None):
    """Embed a user question for searching. Use this at query time, not the above."""
    client = client or get_client()
    return np.asarray(_embed(client, [text], "search_query")[0], dtype=np.float32)


def load_chunks(path=CHUNKS_PATH):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    chunks = load_chunks()
    print(f"{len(chunks)} chunks -> {MODEL} ({DIMENSION}d)")

    vectors = embed_documents([c["text"] for c in chunks])

    np.save(os.path.join(OUT_DIR, "vectors.npy"), vectors)
    with open(os.path.join(OUT_DIR, "ids.json"), "w", encoding="utf-8") as fh:
        json.dump([c["id"] for c in chunks], fh)

    norms = np.linalg.norm(vectors, axis=1)
    print(f"\nsaved {vectors.shape} to {OUT_DIR}")
    print(f"norms min {norms.min():.3f} max {norms.max():.3f} (should be ~1.0)")


if __name__ == "__main__":
    main()
