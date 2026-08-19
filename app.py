import argparse
import os
import sys
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env")

# UPDATE THESE PATHS to match your actual chunk file and vector database locations
CHUNKS_PATH = Path("data/processed/chunks.jsonl")      
VECTOR_DB_PATH = Path("data/chroma")             


def is_index_missing() -> bool:
    """Checks if either the chunk file or vector storage is missing or empty."""
    if not CHUNKS_PATH.exists():
        print(f"[Auto-Index Trigger] Chunk file missing at: {CHUNKS_PATH}")
        return True
        
    if not VECTOR_DB_PATH.exists() or (VECTOR_DB_PATH.is_dir() and not any(VECTOR_DB_PATH.iterdir())):
        print(f"[Auto-Index Trigger] Vector storage missing or empty at: {VECTOR_DB_PATH}")
        return True

    return False


def build_index():
    """Executes the full indexing workflow."""
    from src.indexing.parse import main as run_parse_chunk
    from src.vector_db.collections import build as run_build_index

    print("\n=== CLI ROUTINE: Building Index ===")
    run_parse_chunk()
    run_build_index()
    print("\n=== CLI ROUTINE: Index Completed successfully! ===\n")


def start_server(host: str, port: int):
    print(f"Starting Clinical Hepatology FastAPI server on {host}:{port}...")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(description="Clinical Hepatology RAG System Service & CLI Runner")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API host server address")
    parser.add_argument("--port", type=int, default=8000, help="API server port number")
    parser.add_argument("--index", action="store_true", help="Synchronously build/rebuild the vector index")
    parser.add_argument("--query", type=str, help="Synchronously execute a single RAG query locally")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve for queries")

    args = parser.parse_args()

 
    if is_index_missing():
        build_index()
        start_server(args.host, args.port)
    elif args.query:
        from src.generation.pipeline import run_rag
        print(f"=== CLI ROUTINE: Querying '{args.query}' ===")
        run_rag(args.query, top_k=args.top_k)
    else:
        start_server(args.host, args.port)


if __name__ == "__main__":
    main()