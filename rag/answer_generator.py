"""Generate grounded extractive answers using retrieved context."""

import re

from text_analytics.keyword_extraction import STOP_WORDS
from text_analytics.preprocessing import tokenize


def _content_terms(text: str) -> set[str]:
    """Return terms that convey the subject of a question."""
    return {term for term in tokenize(text) if term not in STOP_WORDS and len(term) > 1}


def _passages(text: str, query_terms: set[str], maximum_words: int = 55) -> list[str]:
    """Split text into answer-sized passages, creating a focused window when needed."""
    passages = [passage.strip() for passage in re.split(r"(?<=[.!?;])\s+|(?=Slide\s+\d+\s*:)", text) if passage.strip()]
    focused = []
    for passage in passages:
        words = passage.split()
        if len(words) <= maximum_words:
            focused.append(passage)
            continue
        matching_positions = [index for index, word in enumerate(words) if set(tokenize(word)) & query_terms]
        if not matching_positions:
            focused.append(" ".join(words[:maximum_words]))
            continue
        best_start = max(0, matching_positions[0] - maximum_words // 3)
        focused.append(" ".join(words[best_start : best_start + maximum_words]))
    return focused


def _definition_question(question: str) -> bool:
    """Identify questions that expect a definition rather than a topic mention."""
    normalized = " ".join(tokenize(question))
    return normalized.startswith(("what is", "what are", "define", "meaning of", "explain what"))


def _definition_bonus(passage: str, question: str) -> float:
    """Prefer passages that define the requested concept over table-of-contents text."""
    concept = [term for term in tokenize(question) if term not in STOP_WORDS and len(term) > 1]
    if not concept:
        return 0.0
    concept_pattern = r"\s+".join(map(re.escape, concept))
    normalized = " ".join(tokenize(passage))
    definition_pattern = rf"\b(?:an?\s+|the\s+)?{concept_pattern}\s+(?:is|are|means|refers\s+to|can\s+be\s+defined\s+as)\b"
    if re.search(r"\b(chapter|chapters|section|sections|this book|we discuss|will discuss|learning objective)\b", normalized):
        return -0.55
    if re.search(definition_pattern, normalized):
        return 0.65
    return -0.15


def _score_passage(passage: str, question: str, query_terms: set[str], retrieval_score: float) -> float:
    """Score a short evidence passage using direct term coverage and rank evidence."""
    terms = set(tokenize(passage))
    coverage = len(query_terms & terms) / len(query_terms) if query_terms else 0.0
    phrase_bonus = 0.15 if len(query_terms) > 1 and len(query_terms & terms) > 1 else 0.0
    definition_bonus = _definition_bonus(passage, question) if _definition_question(question) else 0.0
    return coverage + phrase_bonus + definition_bonus + (0.05 * retrieval_score)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Create a focused, source-grounded answer from the strongest evidence passages."""
    if not chunks:
        return {"answer": "I could not find relevant information in the indexed documents.", "sources": []}

    query_terms = _content_terms(question)
    candidates = []
    for chunk_index, chunk in enumerate(chunks):
        for passage in _passages(chunk["text"], query_terms):
            candidates.append((_score_passage(passage, question, query_terms, chunk.get("score", 0.0)), chunk_index, passage))

    candidates.sort(key=lambda item: item[0], reverse=True)
    answer = candidates[0][2] if candidates else chunks[0]["text"]
    sources = []
    for chunk in chunks:
        filename = chunk["metadata"]["filename"]
        if filename not in {source["document_name"] for source in sources}:
            sources.append({"document_name": filename, "page": None, "score": chunk.get("score")})
    return {"answer": answer, "sources": sources}
