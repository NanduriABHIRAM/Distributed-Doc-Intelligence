"""End-to-end orchestration for the midterm backend prototype."""

from dataclasses import dataclass, field

from chunking.chunker import chunk_documents
from embeddings.generate_embeddings import generate_embeddings
from ingestion.ingestion_pipeline import run_ingestion
from rag.answer_generator import generate_answer
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import rerank
from retrieval.vector_retriever import VectorRetriever


@dataclass
class DocumentIntelligencePipeline:
    """Build indexes once, then answer multiple document questions."""

    source_path: str
    documents: list[dict] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    bm25: BM25Retriever | None = None
    vector: VectorRetriever | None = None
    hybrid: HybridRetriever | None = None

    def build(self) -> "DocumentIntelligencePipeline":
        """Ingest, chunk, embed, and index the configured source directory."""
        self.documents = run_ingestion(self.source_path)
        self.chunks = generate_embeddings(chunk_documents(self.documents))
        self.bm25 = BM25Retriever(self.chunks)
        self.vector = VectorRetriever(self.chunks)
        self.hybrid = HybridRetriever(self.bm25, self.vector)
        return self

    def stats(self) -> dict:
        """Return summary counts for terminal and API display."""
        return {"documents": len(self.documents), "chunks": len(self.chunks), "embeddings": len(self.chunks)}

    def search(self, question: str, top_k: int = 5) -> dict:
        """Run BM25, vector, hybrid retrieval, and reranking for a question."""
        if self.hybrid is None or self.bm25 is None or self.vector is None:
            raise RuntimeError("Build the pipeline before searching.")
        bm25_results = self.bm25.search(question, top_k)
        vector_results = self.vector.search(question, top_k)
        hybrid_results = rerank(question, self.hybrid.search(question, top_k * 2), top_k)
        return {"bm25": bm25_results, "vector": vector_results, "hybrid": hybrid_results}

    def answer(self, question: str, top_k: int = 5) -> dict:
        """Generate a grounded answer and include retrieval details."""
        results = self.search(question, top_k)
        return {"question": question, "retrieval": results, **generate_answer(question, results["hybrid"])}
