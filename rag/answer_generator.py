"""Generate grounded extractive answers using retrieved context."""

import re

from text_analytics.preprocessing import tokenize


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Create a source-grounded extractive answer from the best chunks."""
    if not chunks:
        return {"answer": "I could not find relevant information in the indexed documents.", "sources": []}

    query_terms = set(tokenize(question))
    sentences = re.split(r"(?<=[.!?])\s+", chunks[0]["text"])
    best_sentence = max(sentences, key=lambda sentence: len(query_terms & set(tokenize(sentence))), default=chunks[0]["text"])
    answer = best_sentence.strip() or chunks[0]["text"]
    sources = [
        {"document_name": chunk["metadata"]["filename"], "page": None, "score": chunk.get("score")}
        for chunk in chunks
    ]
    return {"answer": answer, "sources": sources}
