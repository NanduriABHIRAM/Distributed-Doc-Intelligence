"""Lightweight frequency-based keyword extraction."""

from collections import Counter

from text_analytics.preprocessing import tokenize

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or",
    "that", "the", "to", "with", "what", "which", "who", "when", "where", "why", "how", "explain", "define",
    "describe", "tell", "give", "please",
}


def extract_keywords(text: str, top_k: int = 10) -> list[str]:
    """Return frequent non-stopword tokens from *text*."""
    counts = Counter(token for token in tokenize(text) if token not in STOP_WORDS and len(token) > 2)
    return [token for token, _ in counts.most_common(top_k)]
