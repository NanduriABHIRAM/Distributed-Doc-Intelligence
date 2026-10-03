"""End-to-end orchestration for the midterm backend prototype."""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re

# Load .env before importing modules that read DDI_* defaults at import time.
import config  # noqa: F401

from chunking.chunker import chunk_documents  # noqa: F401 - flat chunker kept for A/B comparison
from chunking.structure_chunker import chunk_documents_structured
from embeddings.embedding_cache import describe
from embeddings.generate_embeddings import generate_embeddings
from embeddings.sentence_encoder import load_embedding_model
from ingestion.ingestion_pipeline import run_ingestion
from ingestion.chunk_cache import corpus_fingerprint, load_chunk_cache, save_chunk_cache
from ingestion.processed_chunk_loader import load_processed_chunks
from rag.answer_generator import generate_answer
from rag.extractive_generator import generate_extractive_answer
from rag.llm_client import is_configured as llm_is_configured
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import rerank
from retrieval.vector_retriever import VectorRetriever
from text_analytics.keyword_extraction import STOP_WORDS
from text_analytics.preprocessing import expand_terms, tokenize

logger = logging.getLogger(__name__)


def _definition_question(question: str) -> bool:
    """Return whether *question* asks for a concept definition."""
    normalized = " ".join(tokenize(question))
    return normalized.startswith(("what is", "what are", "define", "meaning of", "explain what"))


def _definition_candidates(question: str, chunks: list[dict], limit: int) -> list[dict]:
    """Find passages phrased as definitions that lexical ranking can overlook."""
    concept = [term for term in tokenize(question) if term not in STOP_WORDS and len(term) > 1]
    if not concept:
        return []
    concept_pattern = r"\s+".join(map(re.escape, concept))
    pattern = re.compile(
        rf"\b(?:an?\s+|the\s+)?{concept_pattern}\s+(?:is|are|means|refers\s+to|can\s+be\s+defined\s+as)\b",
        flags=re.IGNORECASE,
    )
    matches = []
    for chunk in chunks:
        if pattern.search(chunk["text"]):
            # Scored, not flat. A constant score made every definition match
            # tie, so the top of the results was whichever one the chunker
            # happened to emit first rather than the best evidence.
            matches.append(
                dict(
                    chunk,
                    score=0.25 + 0.10 * _concept_coverage(chunk["text"], set(concept)),
                    retrieval_method="definition",
                    methods=["definition"],
                )
            )
            if len(matches) == limit:
                break
    return matches


def _concept_coverage(text: str, concept: set[str]) -> float:
    """Fraction of the queried concept's terms present in *text*."""
    if not concept:
        return 0.0
    present = set(expand_terms(text))
    return len(concept & present) / len(concept)


def _diversify(results: list[dict], top_k: int, per_document: int = 2) -> list[dict]:
    """Limit how many chunks any single document may contribute.

    Retrieval operates on chunks, but a user asking "what is an operating
    system?" wants the defining sentence, not three consecutive chunks from the
    same 900-page textbook. Without a per-document cap the fused ranking is
    dominated by whichever document produced the most chunks, and the actual
    defining slide in a smaller deck never reaches the answer.
    """
    counts: dict[str, int] = {}
    selected: list[dict] = []
    overflow: list[dict] = []
    for result in results:
        filename = result.get("metadata", {}).get("filename", "unknown")
        if counts.get(filename, 0) < per_document:
            counts[filename] = counts.get(filename, 0) + 1
            selected.append(result)
        else:
            overflow.append(result)
        if len(selected) == top_k:
            break
    # If diversity left the list short, backfill with the best remaining chunks
    # rather than returning fewer results than requested.
    for result in overflow:
        if len(selected) >= top_k:
            break
        selected.append(result)
    return selected


@dataclass
class DocumentIntelligencePipeline:
    """Build indexes once, then answer multiple document questions."""

    source_path: str
    processed_chunks_path: str | None = None
    documents: list[dict] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    bm25: BM25Retriever | None = None
    vector: VectorRetriever | None = None
    hybrid: HybridRetriever | None = None
    embedding_model_name: str = "unknown"
    chunk_cache_path: str | None = None

    def build(self) -> "DocumentIntelligencePipeline":
        """Ingest, chunk, embed, and index the configured source directory."""
        if self.processed_chunks_path:
            self.chunks = load_processed_chunks(self.processed_chunks_path)
            document_ids = {chunk["document_id"] for chunk in self.chunks}
            self.documents = [{"document_id": document_id} for document_id in document_ids]
        else:
            fingerprint = corpus_fingerprint(self.source_path)
            cache_path = self.chunk_cache_path or os.environ.get(
                "DDI_CHUNK_CACHE", str(Path("data") / "processed" / "chunks.json")
            )
            cached = load_chunk_cache(cache_path, fingerprint)
            if cached is not None:
                self.documents, self.chunks = cached
                logger.info("Reused %d cached chunks from %s", len(self.chunks), cache_path)
            else:
                self.documents = run_ingestion(self.source_path)
                # Structure-aware chunking: a slide or section heading is a semantic
                # boundary that flat word-count splitting cuts through, which is how
                # the sentence that defines a concept ends up separated from the
                # heading that gives it meaning.
                self.chunks = chunk_documents_structured(self.documents)
                try:
                    save_chunk_cache(cache_path, fingerprint, self.documents, self.chunks)
                except OSError as error:
                    # A cache is an optimization only.  Retrieval remains usable
                    # even if the local directory is read-only or out of space.
                    logger.warning("Could not save chunk cache %s: %s", cache_path, error)
        # The query encoder must be the same model that encoded the chunks.
        embedding_model = load_embedding_model()
        self.embedding_model_name = describe(embedding_model)
        self.chunks = generate_embeddings(self.chunks, embedding_model)
        self.bm25 = BM25Retriever(self.chunks)
        self.vector = VectorRetriever(self.chunks, embedding_model)
        self.hybrid = HybridRetriever(self.bm25, self.vector)
        return self

    def stats(self) -> dict:
        """Return summary counts for terminal and API display."""
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "embeddings": len(self.chunks),
            # Named explicitly because a silent fallback to hashed embeddings
            # produces plausible scores while destroying semantic retrieval.
            "embedding_model": self.embedding_model_name,
        }

    def search(self, question: str, top_k: int = 5) -> dict:
        """Run BM25, vector, hybrid retrieval, and reranking for a question."""
        if self.hybrid is None or self.bm25 is None or self.vector is None:
            raise RuntimeError("Build the pipeline before searching.")
        bm25_results = self.bm25.search(question, top_k)
        vector_results = self.vector.search(question, top_k)
        candidate_count = max(top_k * 10, 20)
        candidates = self.hybrid.search(question, candidate_count)
        if _definition_question(question):
            seen_chunk_ids = {candidate["chunk_id"] for candidate in candidates}
            definition_hits = [
                candidate
                for candidate in _definition_candidates(question, self.chunks, candidate_count)
                if candidate["chunk_id"] not in seen_chunk_ids
            ]
            definition_hits.sort(key=lambda item: item["score"], reverse=True)
            candidates.extend(definition_hits)
        hybrid_results = _diversify(rerank(question, candidates, top_k), top_k)
        return {"bm25": bm25_results, "vector": vector_results, "hybrid": hybrid_results}

    def answer(self, question: str, top_k: int = 5, generator: str = "auto") -> dict:
        """Generate a grounded answer and include retrieval details.

        ``auto`` uses the LLM when credentials are configured and otherwise
        retains extractive answering for local, offline development.  ``llm``
        makes LLM configuration mandatory; ``extractive`` explicitly disables
        the generative step for evaluation comparisons.
        """
        if generator not in {"auto", "llm", "extractive"}:
            raise ValueError("generator must be 'auto', 'llm', or 'extractive'.")
        results = self.search(question, top_k)
        response = {
            "question": question,
            "retrieval": results,
        }
        if generator == "llm" or (generator == "auto" and llm_is_configured()):
            try:
                generated = generate_answer(question, results["hybrid"])
            except Exception as error:  # Network, quota, or provider failures must not end a student session.
                logger.warning("LLM generation failed; using extractive answer: %s", error)
                generated = generate_extractive_answer(question, results["hybrid"])
                generated["generation_warning"] = "Gemini was unavailable, so this answer was extracted from the retrieved documents."
            if generated.get("abstained"):
                extracted = generate_extractive_answer(question, results["hybrid"])
                if not extracted.get("abstained"):
                    extracted["generation_warning"] = "Gemini could not ground an answer, so this answer was extracted from the retrieved documents."
                    generated = extracted
            response.update(generated)
        else:
            response.update(generate_extractive_answer(question, results["hybrid"]))
        return response
