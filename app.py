import argparse
import os
import sys
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env")

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
    
    if args.index:
        from src.indexing.parse import main as run_parse_chunk
        from src.vector_db.collections import build as run_build_index
        print("\n=== CLI ROUTINE: Building Index ===")
        run_parse_chunk()
        run_build_index()
        print("\n=== CLI ROUTINE: Index Completed successfully! ===")
    elif args.query:
        from src.generation.pipeline import run_rag
        print(f"\n=== CLI ROUTINE: Querying '{args.query}' ===")
        run_rag(args.query, top_k=args.top_k)
    else:
        # Default action: run the FastAPI web application
        start_server(args.host, args.port)

if __name__ == "__main__":
    main()