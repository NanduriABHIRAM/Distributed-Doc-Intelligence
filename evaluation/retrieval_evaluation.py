"""Evaluate lexical, semantic, and hybrid retrieval quality."""


def evaluate_retrieval(results: list[str], relevant_ids: set[str], k: int = 5) -> dict:
    """Calculate Precision@K, Recall@K, and reciprocal rank for one query."""
    ranked = results[:k]
    hits = [item for item in ranked if item in relevant_ids]
    first_rank = next((index for index, item in enumerate(ranked, 1) if item in relevant_ids), None)
    return {
        "precision_at_k": len(hits) / k if k else 0.0,
        "recall_at_k": len(hits) / len(relevant_ids) if relevant_ids else 0.0,
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
    }
