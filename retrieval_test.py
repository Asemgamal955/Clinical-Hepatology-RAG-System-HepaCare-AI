import argparse
import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.retriever.retriever import retrieve
from src.retriever.query_parser import (
    expand_clinical_terms,
    keywords_only,
)
from src.evaluation.runner import (
    load_qrels,
    run_evaluation,
    print_report,
)


# ============================================================
# OUTPUT PATHS
# ============================================================

DOCS_DIR = PROJECT_ROOT / "docs"

DOCS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RETRIEVAL_RESULTS_FILE = (
    DOCS_DIR / "retrieval_results.txt"
)

EVALUATION_REPORT_FILE = (
    DOCS_DIR / "evaluation_report.txt"
)


# ============================================================
# FILE HELPERS
# ============================================================

def clear_file(
    file_path: Path,
    title: str,
):
    """
    Clear a file and write a header.
    """

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(title)
        f.write("\n")
        f.write("=" * 90)
        f.write("\n\n")


def save_result(
    text: str,
    file_path: Path,
):
    """
    Append text to a file.
    """

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        file_path,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(str(text))
        f.write("\n")


# ============================================================
# BUILD EVALUATION REPORT
# ============================================================

def build_evaluation_report(
    summary: dict,
) -> str:
    """
    Build the formatted evaluation report as a string.

    Metrics:
        P@K
        R@K
        AP@K
        RR
        Macro Precision
        Macro Recall
        MAP@K
        MRR
    """

    k = summary["k"]
    n = summary["num_queries"]
    col = 52

    lines = []

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    lines.append(
        f"{'=' * 90}"
    )

    lines.append(
        f"  RETRIEVAL EVALUATION REPORT   "
        f"(k={k}, queries={n})"
    )

    lines.append(
        f"{'=' * 90}"
    )

    lines.append(
        f"  {'Query':<{col}}  "
        f"P@{k:<3}  "
        f"R@{k:<3}  "
        f"AP@{k:<3}  "
        f"RR"
    )

    lines.append(
        f"  {'-' * col}  "
        f"-----  "
        f"-----  "
        f"-----  "
        f"-----"
    )

    # --------------------------------------------------------
    # Per-query results
    # --------------------------------------------------------

    for row in summary["per_query"]:

        query = row["query"]

        if len(query) > col:

            q_label = (
                query[:col - 1]
                + "…"
            )

        else:

            q_label = query

        lines.append(
            f"  {q_label:<{col}}"
            f"  {row['precision_at_k']:.3f}"
            f"  {row['recall_at_k']:.3f}"
            f"  {row['ap_at_k']:.3f}"
            f"  {row['rr']:.3f}"
        )

    # --------------------------------------------------------
    # Macro average
    # --------------------------------------------------------

    lines.append("")

    lines.append(
        f"  {'MACRO AVERAGE':<{col}}"
        f"  {summary['mean_precision']:.3f}"
        f"  {summary['mean_recall']:.3f}"
        f"  {summary['map_at_k']:.3f}"
        f"  {summary['mrr']:.3f}"
    )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    lines.append(
        f"{'=' * 90}"
    )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# RETRIEVAL TEST
# ============================================================

def run_test(
    query: str,
    top_k: int = 3,
):
    """
    Parse query, display rewrites,
    run retrieval, and save results.
    """

    output = []

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    output.append(
        "\n" + "=" * 70
    )

    output.append(
        f"RAW USER QUERY: '{query}'"
    )

    output.append(
        "=" * 70
    )

    # --------------------------------------------------------
    # Query expansion
    # --------------------------------------------------------

    try:

        expanded, applied = expand_clinical_terms(
            query
        )

        sparse_q = keywords_only(
            expanded
        )

        output.append(
            "[QUERY EXPANSION]"
        )

        output.append(
            f"  Expansions   : "
            f"{applied or 'None'}"
        )

        output.append(
            f"  BM25 Keywords: "
            f"{sparse_q}"
        )

    except Exception as e:

        output.append(
            f"[Query Expansion Error] {e}"
        )

    output.append(
        "-" * 70
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    try:

        results = retrieve(
            query,
            top_k=top_k,
        )

    except FileNotFoundError as e:

        output.append(
            f"[File Error] {e}"
        )

        final_output = "\n".join(
            output
        )

        print(final_output)

        save_result(
            final_output,
            RETRIEVAL_RESULTS_FILE,
        )

        return

    except Exception as e:

        output.append(
            f"[Retrieval Error] {e}"
        )

        final_output = "\n".join(
            output
        )

        print(final_output)

        save_result(
            final_output,
            RETRIEVAL_RESULTS_FILE,
        )

        return

    # --------------------------------------------------------
    # Empty results
    # --------------------------------------------------------

    if not results:

        output.append(
            "No matching chunks found."
        )

        final_output = "\n".join(
            output
        )

        print(final_output)

        save_result(
            final_output,
            RETRIEVAL_RESULTS_FILE,
        )

        return

    # --------------------------------------------------------
    # Retrieval results
    # --------------------------------------------------------

    for i, res in enumerate(
        results,
        start=1,
    ):

        score = res.get(
            "score",
            0.0,
        )

        section = (
            res.get("section_path")
            or res.get("section")
            or "N/A"
        )

        topic = res.get(
            "topic",
            "N/A",
        )

        text = (
            res.get(
                "text",
                "",
            )
            .replace(
                "\n",
                " ",
            )
            .strip()
        )

        output.append(
            f"\n[{i}] "
            f"Score: {score:.4f} | "
            f"Topic: {topic}"
        )

        output.append(
            f"    Section: {section}"
        )

        if len(text) > 200:

            output.append(
                f"    Text: "
                f"{text[:200]}..."
            )

        else:

            output.append(
                f"    Text: {text}"
            )

        output.append(
            "-" * 70
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    final_output = "\n".join(
        output
    )

    print(
        final_output
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_result(
        final_output,
        RETRIEVAL_RESULTS_FILE,
    )


# ============================================================
# EVALUATION FLOW
# ============================================================

def run_evaluation_flow(
    eval_path: str,
    top_k: int,
):
    """
    Run the complete retrieval evaluation.

    The evaluation report is:
        1. Printed using print_report()
        2. Built as a string
        3. Saved to evaluation_report.txt
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "STARTING RETRIEVAL EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nLoading evaluation dataset from:"
        f"\n  {eval_path}"
    )

    try:

        # ----------------------------------------------------
        # Load qrels
        # ----------------------------------------------------

        qrels = load_qrels(
            eval_path
        )

        print(
            f"\nRunning evaluation on "
            f"{len(qrels)} queries..."
        )

        # ----------------------------------------------------
        # Run evaluation
        # ----------------------------------------------------

        summary = run_evaluation(
            qrels,
            top_k=top_k,
            verbose=True,
        )

        # ----------------------------------------------------
        # Print normal report
        #
        # Your existing print_report() returns None.
        # Therefore we DON'T do:
        #
        # report = print_report(summary)
        #
        # ----------------------------------------------------

        print_report(
            summary
        )

        # ----------------------------------------------------
        # Build report ourselves
        # ----------------------------------------------------

        report = build_evaluation_report(
            summary
        )

        # ----------------------------------------------------
        # Save report
        # ----------------------------------------------------

        with open(
            EVALUATION_REPORT_FILE,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                report
            )

        print(
            "\nEvaluation report saved to:"
        )

        print(
            f"  {EVALUATION_REPORT_FILE}"
        )

    except FileNotFoundError as e:

        error_text = (
            "\n[EVALUATION FILE ERROR]\n"
            f"{e}\n"
        )

        print(
            error_text
        )

        save_result(
            error_text,
            EVALUATION_REPORT_FILE,
        )

    except Exception as e:

        error_text = (
            "\n[EVALUATION ERROR]\n"
            f"{e}\n"
        )

        print(
            error_text
        )

        save_result(
            error_text,
            EVALUATION_REPORT_FILE,
        )


# ============================================================
# SAMPLE QUERIES
# ============================================================

SAMPLE_QUERIES = [

    "what should i eat if my liver is fatty",

    "What is liver hepatitis B?",

    (
        "When should screening for Hepatitis B "
        "virus infection be conducted in pregnant "
        "women according to the USPSTF?"
    ),

    (
        "What are the common risk factors for "
        "contracting Hepatitis B infection in "
        "adults in the United States?"
    ),

    (
        "What primary screening test is used to "
        "detect maternal Hepatitis B virus "
        "infection?"
    ),

    (
        "Which age group is recommended by the "
        "USPSTF for Hepatitis C virus screening?"
    ),

    (
        "What is the single most important risk "
        "factor for Hepatitis C virus infection "
        "in the United States?"
    ),

    (
        "What are the standard medications or "
        "interventions used to treat chronic "
        "Hepatitis C infection?"
    ),

    (
        "What underlying physical defect defines "
        "Alagille syndrome within the liver?"
    ),

    (
        "What primary genetic mutations are "
        "responsible for causing Alagille syndrome?"
    ),

    (
        "What physical signs and symptoms in the "
        "body might indicate a diagnosis of "
        "Alagille syndrome?"
    ),

    (
        "What dietary changes or nutritional plans "
        "are recommended for individuals with "
        "Alagille syndrome?"
    ),
]


# ============================================================
# INTERACTIVE MENU
# ============================================================

def interactive_menu(
    top_k: int = 5,
):

    while True:

        print(
            "\n" + "=" * 50
        )

        print(
            "            RETRIEVAL TEST MENU"
        )

        print(
            "=" * 50
        )

        print(
            "  1. Run default sample queries"
        )

        print(
            "  2. Enter a custom search query"
        )

        print(
            "  3. Run retrieval evaluation (benchmark)"
        )

        print(
            "  4. Exit"
        )

        print(
            "-" * 50
        )

        choice = input(
            "Select an option (1-4): "
        ).strip()

        # ====================================================
        # CASE 1
        # ====================================================

        if choice == "1":

            print(
                "\nRunning retrieval tests "
                "on sample queries..."
            )

            # Start a new retrieval result file
            clear_file(
                RETRIEVAL_RESULTS_FILE,
                "RETRIEVAL TEST RESULTS",
            )

            for query in SAMPLE_QUERIES:

                run_test(
                    query,
                    top_k=top_k,
                )

            print(
                "\nAll retrieval results saved to:"
            )

            print(
                f"  {RETRIEVAL_RESULTS_FILE}"
            )

        # ====================================================
        # CASE 2
        # ====================================================

        elif choice == "2":

            query = input(
                "\nEnter your search query: "
            ).strip()

            if not query:

                print(
                    "Query cannot be empty."
                )

                continue

            run_test(
                query,
                top_k=top_k,
            )

            print(
                "\nResult saved to:"
            )

            print(
                f"  {RETRIEVAL_RESULTS_FILE}"
            )

        # ====================================================
        # CASE 3
        # ====================================================

        elif choice == "3":

            eval_path = (
                "data/eval/queries.jsonl"
            )

            run_evaluation_flow(
                eval_path,
                top_k=5,
            )

        # ====================================================
        # CASE 4
        # ====================================================

        elif (
            choice == "4"
            or choice.lower() == "exit"
        ):

            print(
                "\nExiting..."
            )

            print(
                "\nOutput files:"
            )

            print(
                f"  Retrieval : "
                f"{RETRIEVAL_RESULTS_FILE}"
            )

            print(
                f"  Evaluation: "
                f"{EVALUATION_REPORT_FILE}"
            )

            break

        # ====================================================
        # INVALID
        # ====================================================

        else:

            print(
                "\nInvalid selection."
                " Please choose between 1 and 4."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Test retrieval and run evaluation."
        )
    )

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help=(
            "Search query string to test"
        ),
    )

    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of results to return "
            "(default: 5)"
        ),
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    parser.add_argument(
        "--eval",
        nargs="?",
        const="data/eval/queries.jsonl",
        help=(
            "Run full evaluation using the "
            "specified JSONL qrels file "
            "(default: "
            "data/eval/queries.jsonl)"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # EVALUATION MODE
    # ========================================================

    if args.eval:

        run_evaluation_flow(
            args.eval,
            top_k=args.top_k,
        )

    # ========================================================
    # SINGLE QUERY MODE
    # ========================================================

    elif args.query:

        run_test(
            args.query,
            top_k=args.top_k,
        )

        print(
            "\nResult saved to:"
        )

        print(
            f"  {RETRIEVAL_RESULTS_FILE}"
        )

    # ========================================================
    # INTERACTIVE MODE
    # ========================================================

    else:

        interactive_menu(
            top_k=args.top_k
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()