"""Lightweight token-overlap reranker."""

from text_analytics.keyword_extraction import STOP_WORDS
from text_analytics.preprocessing import tokenize


def _content_terms(text: str) -> set[str]:
    """Keep question terms that carry retrieval meaning."""
    return {term for term in tokenize(text) if term not in STOP_WORDS and len(term) > 1}


def _phrase_matches(query_terms: list[str], candidate_terms: list[str]) -> int:
    """Count adjacent query-term pairs that also occur in the candidate."""
    if len(query_terms) < 2:
        return 0
    candidate_pairs = set(zip(candidate_terms, candidate_terms[1:]))
    return sum(pair in candidate_pairs for pair in zip(query_terms, query_terms[1:]))


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Prioritize chunks that directly cover meaningful question terms and phrases."""
    query_term_list = [term for term in tokenize(query) if term not in STOP_WORDS and len(term) > 1]
    query_terms = set(query_term_list)
    ranked = []
    for candidate in candidates:
        candidate_term_list = tokenize(candidate["text"])
        terms = set(candidate_term_list)
        coverage = len(query_terms & terms) / len(query_terms) if query_terms else 0.0
        phrase_count = _phrase_matches(query_term_list, candidate_term_list)
        phrase_coverage = phrase_count / max(len(query_term_list) - 1, 1)
        keyword_overlap = len(query_terms & set(candidate.get("keywords", []))) / len(query_terms) if query_terms else 0.0
        score = candidate["score"] + (0.60 * coverage) + (0.20 * phrase_coverage) + (0.10 * keyword_overlap)
        ranked.append(
            dict(
                candidate,
                score=round(score, 6),
                rerank_coverage=round(coverage, 3),
                phrase_coverage=round(phrase_coverage, 3),
            )
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k]
