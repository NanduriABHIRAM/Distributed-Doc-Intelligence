"""In-memory vector similarity retriever."""

from embeddings.sentence_encoder import load_embedding_model


class VectorRetriever:
    """Retrieve chunks with cosine similarity over normalized embeddings."""

    def __init__(self, chunks: list[dict] | None = None, model=None):
        self.model = model or load_embedding_model()
        self.chunks: list[dict] = []
        if chunks:
            self.index(chunks)

    def index(self, chunks: list[dict]) -> None:
        """Store chunks that have already been embedded."""
        missing = [chunk["chunk_id"] for chunk in chunks if "embedding" not in chunk]
        if missing:
            raise ValueError("All chunks must have an embedding before vector indexing")
        self.chunks = chunks

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return chunks most similar to *query*."""
        if not self.chunks or not query.strip():
            return []
        query_vector = self.model.encode(query)
        scores = []
        for chunk in self.chunks:
            score = sum(left * right for left, right in zip(query_vector, chunk["embedding"]))
            if score > 0:
                scores.append((score, chunk))
        return [dict(chunk, score=round(score, 6), retrieval_method="vector") for score, chunk in sorted(scores, key=lambda item: item[0], reverse=True)[:top_k]]
