"""In-memory BM25 lexical retriever."""

from collections import Counter
from math import log

from text_analytics.preprocessing import tokenize


class BM25Retriever:
    """Retrieve chunks using the Okapi BM25 ranking formula."""

    def __init__(self, chunks: list[dict] | None = None, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[dict] = []
        self.term_frequencies: list[Counter] = []
        self.document_frequencies: Counter = Counter()
        self.average_length = 0.0
        if chunks:
            self.index(chunks)

    def index(self, chunks: list[dict]) -> None:
        """Build a lexical index over *chunks*."""
        self.chunks = chunks
        self.term_frequencies = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        self.document_frequencies = Counter()
        for frequencies in self.term_frequencies:
            self.document_frequencies.update(frequencies.keys())
        self.average_length = sum(sum(freq.values()) for freq in self.term_frequencies) / len(chunks) if chunks else 0.0

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top lexical matches for *query*."""
        if not self.chunks or not query.strip():
            return []
        total_documents = len(self.chunks)
        scores = []
        for chunk, frequencies in zip(self.chunks, self.term_frequencies):
            length = sum(frequencies.values())
            score = 0.0
            for term in tokenize(query):
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                idf = log(1 + (total_documents - self.document_frequencies[term] + 0.5) / (self.document_frequencies[term] + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average_length)
                score += idf * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scores.append((score, chunk))
        return [dict(chunk, score=round(score, 6), retrieval_method="bm25") for score, chunk in sorted(scores, key=lambda item: item[0], reverse=True)[:top_k]]
