"""Lightweight token-overlap reranker."""

from text_analytics.preprocessing import tokenize


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Boost hybrid results that directly cover important query terms."""
    query_terms = set(tokenize(query))
    ranked = []
    for candidate in candidates:
        terms = set(tokenize(candidate["text"]))
        coverage = len(query_terms & terms) / len(query_terms) if query_terms else 0.0
        score = candidate["score"] + (0.05 * coverage)
        ranked.append(dict(candidate, score=round(score, 6), rerank_coverage=round(coverage, 3)))
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k]
