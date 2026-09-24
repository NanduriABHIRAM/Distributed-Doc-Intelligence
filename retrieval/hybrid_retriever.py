"""Hybrid retrieval using Reciprocal Rank Fusion."""

from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_retriever import VectorRetriever


class HybridRetriever:
    """Fuse BM25 and vector rankings with Reciprocal Rank Fusion (RRF)."""

    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever, rrf_k: int = 60):
        self.bm25 = bm25
        self.vector = vector
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return fused lexical and semantic matches for *query*."""
        candidates: dict[str, dict] = {}
        for method, results in (("bm25", self.bm25.search(query, top_k * 3)), ("vector", self.vector.search(query, top_k * 3))):
            for rank, result in enumerate(results, start=1):
                item = candidates.setdefault(result["chunk_id"], dict(result, score=0.0, retrieval_method="hybrid", methods=[]))
                item["score"] += 1 / (self.rrf_k + rank)
                item["methods"].append(method)
        ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)[:top_k]
        return [dict(item, score=round(item["score"], 6)) for item in ranked]
