import argparse
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.retriever.retriever import retrieve
from src.retriever.query_parser import expand_clinical_terms, keywords_only
from src.evaluation.runner import load_qrels, run_evaluation, print_report


def run_test(query: str, top_k: int = 3):
    """Parse query, display rewrites, and run retrieval."""
    print(f"\n" + "=" * 70)
    print(f"RAW USER QUERY: '{query}'")
    print("=" * 70)

    # Show clinical term expansions and keyword form
    expanded, applied = expand_clinical_terms(query)
    sparse_q = keywords_only(expanded)
    print("[QUERY EXPANSION]")
    print(f"  Expansions   : {applied or 'None'}")
    print(f"  BM25 Keywords: {sparse_q}")
    print("-" * 70)

    # 2. Execute retrieval
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


def run_evaluation_flow(eval_path: str, top_k: int):
    print(f"\nLoading evaluation dataset from: {eval_path}...")
    try:
        qrels = load_qrels(eval_path)
        print(f"Running evaluation on {len(qrels)} queries...")
        summary = run_evaluation(qrels, top_k=top_k, verbose=True)
        print_report(summary)
    except Exception as e:
        print(f"\n[Evaluation Error] {e}")


def interactive_menu(top_k: int = 5):
    while True:
        print("\n" + "=" * 50)
        print("            RETRIEVAL TEST MENU")
        print("=" * 50)
        print("  1. Run default sample queries")
        print("  2. Enter a custom search query")
        print("  3. Run retrieval evaluation (benchmark)")
        print("  4. Exit")
        print("-" * 50)
        
        choice = input("Select an option (1-4): ").strip()
        
        if choice == "1":
            sample_queries = [
                "what should i eat if my liver is fatty",
                "What is liver hepatitis B?",
                "When should screening for Hepatitis B virus infection be conducted in pregnant women according to the USPSTF?",
                "What are the common risk factors for contracting Hepatitis B infection in adults in the United States?",
                "What primary screening test is used to detect maternal Hepatitis B virus infection?",
                "Which age group is recommended by the USPSTF for Hepatitis C virus screening?",
                "What is the single most important risk factor for Hepatitis C virus infection in the United States?",
                "What are the standard medications or interventions used to treat chronic Hepatitis C infection?",
                "What underlying physical defect defines Alagille syndrome within the liver?",
                "What primary genetic mutations are responsible for causing Alagille syndrome?",
                "What physical signs and symptoms in the body might indicate a diagnosis of Alagille syndrome?",
                "What dietary changes or nutritional plans are recommended for individuals with Alagille syndrome?"
            ]
            print("\nRunning retrieval tests on sample queries...")
            for query in sample_queries:
                run_test(query, top_k=top_k)
        elif choice == "2":
            query = input("\nEnter your search query: ").strip()
            if query:
                run_test(query, top_k=top_k)
            else:
                print("Query cannot be empty.")
        elif choice == "3":
            eval_path = "data/eval/queries.jsonl"
            if not eval_path:
                eval_path = "data/eval/queries.jsonl"
            run_evaluation_flow(eval_path, top_k=5)
        elif choice == "4" or choice.lower() == "exit":
            print("Exiting...")
            break
        else:
            print("Invalid selection. Please choose between 1 and 4.")

def main():
    parser = argparse.ArgumentParser(description="Test retrieval and run evaluation.")
    parser.add_argument("query", type=str, nargs="?", help="Search query string to test")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument(
        "--eval",
        nargs="?",
        const="data/eval/queries.jsonl",
        help="Run full evaluation using the specified JSONL qrels file (default: data/eval/queries.jsonl)",
    )
    args = parser.parse_args()

    if args.eval:
        run_evaluation_flow(args.eval, top_k=args.top_k)
    elif args.query:
        run_test(args.query, top_k=args.top_k)
    else:
        # No CLI arguments provided, display interactive menu
        interactive_menu(top_k=args.top_k)


if __name__ == "__main__":
    main()