# retrieval_test.py
import argparse
import sys
from pathlib import Path
from src.retriever.retriever import retrieve
# Ensure root directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parent))


def run_test(query: str, top_k: int = 5):
    """Run a single retrieval test and display results."""
    print(f"\n" + "=" * 70)
    print(f"QUERY: '{query}' (top_k={top_k})")
    print("=" * 70)

    try:
        results = retrieve(query, top_k=top_k)
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        return

    if not results:
        print("No matching chunks found.")
        return

    for i, res in enumerate(results, 1):
        score = res.get("score", 0.0)
        section = res.get("section_path") or res.get("section") or "N/A"
        topic = res.get("topic", "N/A")
        text = res.get("text", "").replace("\n", " ").strip()

        print(f"\n[{i}] Score: {score:.4f} | Topic: {topic}")
        print(f"    Section: {section}")
        print(f"    Text: {text[:200]}..." if len(text) > 200 else f"    Text: {text}")
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test script for vector retrieval outside src.")
    parser.add_argument("query", type=str, nargs="?", help="Search query string")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    args = parser.parse_args()

    # Use CLI query if provided, otherwise run default benchmark test queries
    if args.query:
        run_test(args.query, top_k=args.top_k)
    else:
        print("No query provided via CLI. Running automated benchmark queries...")
        sample_queries = [
            "What is liver hepatitis A?",
            "What is liver hepatitis B?"
        ]
        for q in sample_queries:
            run_test(q, top_k=args.top_k)


if __name__ == "__main__":
    main()