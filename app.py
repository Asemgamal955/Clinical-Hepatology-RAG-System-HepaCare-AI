# main.py
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Import pipeline components
from src.indexing.parse_chunk import main as run_parse_chunk, OUT_PATH as CHUNKS_PATH
from src.indexing.embeddings import main as run_embeddings, embed_query, load_chunks
from src.generation import Gemini

VECTORS_PATH = os.path.join("data", "embeddings", "vectors.npy")
IDS_PATH = os.path.join("data", "embeddings", "ids.json")


# ==========================================
# 1. Pipeline Indexing Routine
# ==========================================

def index_pipeline():
    """Run end-to-end processing from raw files to embedded vectors."""
    print("=== Step 1: Parsing & Chunking Raw Data ===")
    run_parse_chunk()

    print("\n=== Step 2: Generating Cohere Embeddings ===")
    run_embeddings()
    print("\nIndexing completed successfully!")


# ==========================================
# 2. Vector Search / Retrieval Engine
# ==========================================

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve top-k most relevant chunks for a given user query."""
    if not os.path.exists(VECTORS_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "Index files missing! Run `python main.py --index` first to build the search database."
        )

    # Load pre-indexed data
    chunks = load_chunks(CHUNKS_PATH)
    vectors = np.load(VECTORS_PATH)

    # Embed query vector
    query_vec = embed_query(query)

    # Compute cosine similarity (dot product on normalized vectors)
    similarities = np.dot(vectors, query_vec)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["score"] = float(similarities[idx])
        results.append(chunk)

    return results


# ==========================================
# 3. RAG Pipeline & Generation
# ==========================================

def run_rag(query: str, top_k: int = 4, stream: bool = True):
    """Execute full RAG generation pipeline."""
    print(f"\n[Query]: {query}\n")
    print(f"Retrieving top {top_k} contexts...")
    retrieved_chunks = retrieve(query, top_k=top_k)

    # Build context block utilizing section hierarchy
    context_blocks = []
    for i, c in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"--- Context [{i}] (Score: {c['score']:.3f}) ---\n"
            f"Source: {c['url']} | Path: {c['section_path']}\n"
            f"Content:\n{c['text']}"
        )
    context_str = "\n\n".join(context_blocks)

    system_instruction = (
        "You are an expert technical AI assistant. Answer the user's question accurately "
        "and concisely using solely the provided context chunks. Cite sources or section "
        "paths where appropriate. If the answer cannot be determined from context, state so."
    )

    prompt = f"Context:\n{context_str}\n\nUser Question: {query}\n\nAnswer:"

    llm = Gemini(model="gemini-2.5-flash", system_instruction=system_instruction)

    print("\n[Gemini Answer]:")
    if stream:
        for chunk in llm.generate_stream(prompt):
            print(chunk, end="", flush=True)
        print("\n")
    else:
        response = llm.generate(prompt)
        print(response)


# ==========================================
# 4. CLI Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="End-to-End RAG System with Gemini & Cohere")
    parser.add_argument("--index", action="store_true", help="Parse data and generate vector index")
    parser.add_argument("--query", type=str, help="Run RAG query against stored knowledge base")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")

    args = parser.parse_args()

    if args.index:
        index_pipeline()
    elif args.query:
        run_rag(args.query, top_k=args.top_k)
    else:
        # Interactive loop if no flags supplied
        if not os.path.exists(VECTORS_PATH):
            print("No index detected. Building vector index now...")
            index_pipeline()

        print("\n=== Interactive RAG System ready! (Type 'exit' to quit) ===")
        while True:
            try:
                user_input = input("\nAsk a question > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break
                run_rag(user_input, top_k=args.top_k)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()