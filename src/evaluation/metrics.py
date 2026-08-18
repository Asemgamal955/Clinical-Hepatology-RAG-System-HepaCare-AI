"""
Pure, stateless IR metric functions.

All functions share the same input contract:
    retrieved_ids : list[str]  - ordered list of chunk["id"] values as returned
                                 by retrieve(), highest-score first.
    relevant_ids  : set[str]   - ground-truth set of chunk IDs that are relevant
                                 for this query (from queries.jsonl).
    k             : int        - rank cut-off to evaluate at.

No network calls, no file I/O - every function takes plain Python scalars /
collections and returns a float. This makes the module trivially unit-testable.

Run the self-test with:  python -m src.evaluation.metrics
"""


# ==============================================================================
# Precision@K
# ==============================================================================

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Fraction of the top-k retrieved items that are relevant.

    P@K = |{retrieved[:k]} ∩ relevant| / k

    Interpretation: "Of the k results shown to the user, how many are actually
    useful?" Ranges [0, 1]; penalises irrelevant items in the top-k window.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


# ==============================================================================
# Recall@K
# ==============================================================================

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Fraction of all relevant items that appear within the top-k results.

    R@K = |{retrieved[:k]} ∩ relevant| / |relevant|

    Interpretation: "Of everything in the knowledge base that could answer this
    query, how much did we surface in the top-k?" Ranges [0, 1]; penalises
    missing relevant chunks regardless of rank.
    Returns 0.0 when relevant_ids is empty to avoid ZeroDivisionError.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


# ==============================================================================
# MAP@K  (Mean Average Precision)
# ==============================================================================

def average_precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Average Precision for a single query at rank cut-off k.

    AP@K = (1 / min(k, |relevant|)) * Σ_{i=1}^{k}  P@i * rel(i)

    where rel(i) = 1 if the item at rank i is relevant, else 0.

    Interpretation: rewards ranking relevant items higher - a hit at rank 1
    contributes more than a hit at rank k. Ranges [0, 1].

    The divisor is min(k, |relevant|), not |relevant|. Normalising by the full
    relevant count makes the metric unreachable whenever a query has more
    relevant chunks than k: with 8 relevant chunks and k=5, a perfect ranking
    would still score 0.625, which reads as a failure when nothing failed.
    Recall@K already measures how much was missed; AP@K should measure ordering.
    The two are identical whenever |relevant| <= k, which is the common case.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    score = 0.0
    hits_so_far = 0

    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            hits_so_far += 1
            score += hits_so_far / i  # precision at this rank

    return score / min(k, len(relevant_ids))


def mean_average_precision(
    results_list: list[tuple[list[str], set[str]]],
    k: int,
) -> float:
    """
    Mean Average Precision across multiple queries at rank cut-off k.

    MAP@K = (1 / |Q|) * Σ_{q} AP@K(q)
    """
    if not results_list:
        return 0.0
    ap_scores = [average_precision_at_k(ret, rel, k) for ret, rel in results_list]
    return sum(ap_scores) / len(ap_scores)


# ==============================================================================
# MRR  (Mean Reciprocal Rank)
# ==============================================================================

def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str], k: int = None) -> float:
    """
    Reciprocal Rank for a single query.

    RR = 1 / rank_of_first_relevant_item

    Interpretation: how quickly the system surfaces *any* correct answer. First
    hit at rank 1 -> 1.0, rank 2 -> 0.5, rank 4 -> 0.25.
    Returns 0.0 if no relevant item appears.

    `k` is optional. Left as None the whole retrieved list is scanned, which
    reports a hit the user would never see if the UI only shows k results.
    Pass k to measure what is actually displayed.
    """
    ids = retrieved_ids if k is None else retrieved_ids[:k]
    for rank, doc_id in enumerate(ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    results_list: list[tuple[list[str], set[str]]],
    k: int = None,
) -> float:
    """
    Mean Reciprocal Rank across multiple queries.

    MRR = (1 / |Q|) * Σ_{q} RR(q)
    """
    if not results_list:
        return 0.0
    rr_scores = [reciprocal_rank(ret, rel, k) for ret, rel in results_list]
    return sum(rr_scores) / len(rr_scores)


# ==============================================================================
# Aggregation helpers
# ==============================================================================

def evaluate_single(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> dict:
    """Compute all four metrics for a single query."""
    return {
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
        "ap_at_k": average_precision_at_k(retrieved_ids, relevant_ids, k),
        "rr": reciprocal_rank(retrieved_ids, relevant_ids, k),
        "k": k,
    }


def aggregate(results_list: list[tuple[list[str], set[str]]], k: int) -> dict:
    """
    Corpus-level metrics over every query.

    Returns Precision@K, Recall@K, MAP@K, MRR@K and the query count, all
    macro-averaged (each query weighted equally regardless of how many
    relevant chunks it has).
    """
    if not results_list:
        return {"queries": 0, "precision_at_k": 0.0, "recall_at_k": 0.0,
                "map_at_k": 0.0, "mrr": 0.0, "k": k}

    n = len(results_list)
    return {
        "queries": n,
        "precision_at_k": sum(precision_at_k(r, g, k) for r, g in results_list) / n,
        "recall_at_k": sum(recall_at_k(r, g, k) for r, g in results_list) / n,
        "map_at_k": mean_average_precision(results_list, k),
        "mrr": mean_reciprocal_rank(results_list, k),
        "k": k,
    }


# ==============================================================================
# Self-test
# ==============================================================================

if __name__ == "__main__":
    def check(label, got, want, tol=1e-6):
        status = "PASS" if abs(got - want) < tol else "FAIL"
        print(f"  {status}  {label:<26} {got:.4f}  (expected {want:.4f})")
        return status == "PASS"

    passed = []

    # retrieved = [a, b, c, d]  relevant = {a, c}  k = 4
    #   P@4  = 2/4 = 0.5
    #   R@4  = 2/2 = 1.0
    #   AP@4 = (1/1 + 2/3) / 2 = 0.8333
    #   RR   = 1/1 = 1.0
    print("=== case 1: hit at rank 1 and 3 ===")
    m = evaluate_single(["a", "b", "c", "d"], {"a", "c"}, 4)
    passed += [check("Precision@4", m["precision_at_k"], 0.5),
               check("Recall@4", m["recall_at_k"], 1.0),
               check("AP@4", m["ap_at_k"], 5 / 6),
               check("RR", m["rr"], 1.0)]

    print("\n=== case 2: no relevant retrieved ===")
    m = evaluate_single(["x", "y"], {"a"}, 2)
    passed += [check("Precision@2", m["precision_at_k"], 0.0),
               check("Recall@2", m["recall_at_k"], 0.0),
               check("AP@2", m["ap_at_k"], 0.0),
               check("RR", m["rr"], 0.0)]

    print("\n=== case 3: perfect ranking ===")
    m = evaluate_single(["a", "b", "c"], {"a", "b"}, 3)
    passed += [check("Precision@3", m["precision_at_k"], 2 / 3),
               check("Recall@3", m["recall_at_k"], 1.0),
               check("AP@3", m["ap_at_k"], 1.0),
               check("RR", m["rr"], 1.0)]

    print("\n=== case 4: more relevant than k (AP stays reachable) ===")
    # 8 relevant, k=3, top 3 all relevant -> ordering is perfect, AP should be 1.0
    m = evaluate_single(["a", "b", "c"], {"a", "b", "c", "d", "e", "f", "g", "h"}, 3)
    passed += [check("Precision@3", m["precision_at_k"], 1.0),
               check("Recall@3", m["recall_at_k"], 3 / 8),
               check("AP@3", m["ap_at_k"], 1.0),
               check("RR", m["rr"], 1.0)]

    print("\n=== case 5: edge cases ===")
    passed += [check("empty relevant", recall_at_k(["a"], set(), 5), 0.0),
               check("k = 0", precision_at_k(["a"], {"a"}, 0), 0.0),
               check("empty retrieved", reciprocal_rank([], {"a"}), 0.0),
               check("MRR of no queries", mean_reciprocal_rank([]), 0.0),
               check("hit below k ignored", reciprocal_rank(["x", "a"], {"a"}, k=1), 0.0)]

    print("\n=== corpus aggregate ===")
    agg = aggregate([(["a", "b"], {"a"}), (["x", "c"], {"c"})], k=2)
    passed += [check("MRR", agg["mrr"], (1.0 + 0.5) / 2),
               check("MAP@2", agg["map_at_k"], (1.0 + 0.5) / 2)]

    print(f"\n{sum(passed)}/{len(passed)} assertions passed")
    assert all(passed), "self-test failed"
