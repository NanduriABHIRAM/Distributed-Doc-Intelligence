"""End-to-end retrieval benchmark across configurable pipeline variants.

Runs a labelled query set against several retrieval strategies and reports
Precision@K, Recall@K, MRR, MAP, and nDCG@K so that changes to individual
pipeline stages can be attributed rather than guessed at.

Usage:
    python -m evaluation.benchmark --variant baseline
    python -m evaluation.benchmark --variant all
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from chunking.chunker import chunk_documents
from embeddings.generate_embeddings import generate_embeddings
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import rerank
from retrieval.vector_retriever import VectorRetriever

DEFAULT_QUERIES = Path(__file__).with_name("labelled_queries.json")
DEFAULT_CORPUS = Path(__file__).with_name("corpus_cache.json")


def resolve_queries(raw: list[dict], corpus: list[dict]) -> list[dict]:
    """Resolve filename patterns to document IDs present in the corpus.

    Labels are stored as filename substrings so they remain readable and stable
    across re-ingestion. Queries whose patterns match nothing are dropped with a
    warning rather than silently scored as unanswerable.
    """
    filenames = {document["document_id"]: document["metadata"].get("filename", "") for document in corpus}
    resolved: list[dict] = []
    dropped: list[str] = []
    for entry in raw:
        patterns = [pattern.lower() for pattern in entry["relevant_filename_patterns"]]
        matched = [
            document_id
            for document_id, filename in filenames.items()
            if any(pattern in filename.lower() for pattern in patterns)
        ]
        if not matched:
            dropped.append(entry["query"])
            continue
        resolved.append(
            {
                "query": entry["query"],
                "type": entry.get("type", "unknown"),
                "relevant_document_ids": sorted(matched),
            }
        )
    for query in dropped:
        print(f"  warning: no corpus match for {query!r}; query skipped")
    return resolved


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not k:
        return 0.0
    return sum(1 for item in ranked[:k] if item in relevant) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in ranked[:k] if item in relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            total += hits / position
    return total / min(len(relevant), len(ranked)) if ranked else 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Normalized discounted cumulative gain using binary relevance."""
    if not relevant:
        return 0.0
    gain = sum(
        1.0 / math_log2(position + 1)
        for position, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal = sum(1.0 / math_log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return gain / ideal if ideal else 0.0


def math_log2(value: float) -> float:
    from math import log2

    return log2(value)


def evaluate_run(ranked: list[str], relevant: set[str], k: int = 5) -> dict[str, float]:
    """Compute the full metric set for one query."""
    return {
        "precision_at_k": round(precision_at_k(ranked, relevant, k), 4),
        "recall_at_k": round(recall_at_k(ranked, relevant, k), 4),
        "reciprocal_rank": round(reciprocal_rank(ranked, relevant), 4),
        "average_precision": round(average_precision(ranked, relevant), 4),
        "ndcg_at_k": round(ndcg_at_k(ranked, relevant, k), 4),
    }


def mean_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Average metrics across queries."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {key: round(statistics.fmean(row[key] for row in per_query), 4) for key in keys}


# --------------------------------------------------------------------------
# Corpus and queries
# --------------------------------------------------------------------------


def load_corpus(path: Path = DEFAULT_CORPUS) -> list[dict]:
    """Load a cached corpus, normalizing the text field to `clean_text`.

    The on-disk cache stores cleaned text under `text` to stay close to the
    raw ingestion record; chunkers expect `clean_text`.
    """
    with open(path, encoding="utf-8") as handle:
        corpus = json.load(handle)
    for document in corpus:
        document.setdefault("clean_text", document.get("text", ""))
    return corpus


def load_queries(path: Path = DEFAULT_QUERIES) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["queries"] if isinstance(payload, dict) else payload


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------


@dataclass
class Variant:
    """A named retrieval configuration to benchmark."""

    name: str
    build_chunks: object
    use_embeddings: bool = True
    use_hybrid: bool = True
    use_rerank: bool = True
    notes: str = ""


def _rank_document_ids(results: list[dict]) -> list[str]:
    """Collapse ranked chunks to a ranked, de-duplicated document list."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in results:
        document_id = item.get("document_id")
        if document_id and document_id not in seen:
            seen.add(document_id)
            ordered.append(document_id)
    return ordered


def build_variant(variant: Variant, corpus: list[dict]) -> dict:
    """Construct the retrievers required by *variant*."""
    chunks = variant.build_chunks(corpus)
    chunks = generate_embeddings(chunks)
    built: dict = {"chunks": chunks, "bm25": BM25Retriever(chunks), "vector": VectorRetriever(chunks)}
    if variant.use_hybrid:
        built["hybrid"] = HybridRetriever(built["bm25"], built["vector"])
    return built


def retrieve(variant: Variant, built: dict, query: str, top_k: int) -> list[str]:
    """Return ranked document IDs for *query* under *variant*."""
    candidate_count = max(top_k * 10, 20)
    if variant.use_hybrid:
        candidates = built["hybrid"].search(query, candidate_count)
    elif variant.use_embeddings:
        candidates = built["vector"].search(query, candidate_count)
    else:
        candidates = built["bm25"].search(query, candidate_count)
    if variant.use_rerank:
        candidates = rerank(query, candidates, candidate_count)
    return _rank_document_ids(candidates)


def long_context_baseline(corpus: list[dict], query: str, top_k: int) -> list[str]:
    """Rank whole documents by term overlap instead of chunk retrieval.

    This approximates a long-context setup where the top documents are supplied
    whole, and exists to test whether chunk-level retrieval earns its cost on
    this corpus at all.
    """
    from text_analytics.keyword_extraction import STOP_WORDS
    from text_analytics.preprocessing import tokenize

    query_terms = {term for term in tokenize(query) if term not in STOP_WORDS and len(term) > 1}
    if not query_terms:
        return []
    scored = []
    for document in corpus:
        document_terms = set(tokenize(document["text"]))
        coverage = len(query_terms & document_terms) / len(query_terms)
        if coverage > 0:
            scored.append((coverage, document["document_id"]))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [document_id for _, document_id in scored[:top_k]]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_benchmark(variants: list[Variant], corpus: list[dict], queries: list[dict], top_k: int = 5) -> dict:
    """Run every variant over every query and return aggregated results."""
    report: dict = {"corpus_documents": len(corpus), "queries": len(queries), "top_k": top_k, "results": {}}

    for variant in variants:
        built = build_variant(variant, corpus)
        per_query = []
        for entry in queries:
            relevant = set(entry["relevant_document_ids"])
            ranked = retrieve(variant, built, entry["query"], top_k)
            per_query.append(evaluate_run(ranked, relevant, top_k))
        report["results"][variant.name] = {
            "aggregate": mean_metrics(per_query),
            "chunks_indexed": len(built["chunks"]),
            "notes": variant.notes,
        }
        print(f"  {variant.name}: {report['results'][variant.name]['aggregate']}")

    baseline_rows = [evaluate_run(long_context_baseline(corpus, entry["query"], top_k), set(entry["relevant_document_ids"]), top_k) for entry in queries]
    report["results"]["long_context_document_baseline"] = {
        "aggregate": mean_metrics(baseline_rows),
        "chunks_indexed": 0,
        "notes": "Ranks whole documents by term overlap; no chunking, no embeddings.",
    }
    print(f"  long_context_document_baseline: {report['results']['long_context_document_baseline']['aggregate']}")
    return report


def default_variants() -> list[Variant]:
    """Variants ordered so each isolates one pipeline stage."""
    from chunking.structure_chunker import chunk_documents_structured

    return [
        Variant("bm25_only", lambda corpus: chunk_documents(corpus), use_embeddings=False, use_hybrid=False, use_rerank=False,
                notes="Flat chunking, lexical only. Isolates the lexical floor."),
        Variant("vector_hash_only", lambda corpus: chunk_documents(corpus), use_embeddings=True, use_hybrid=False, use_rerank=False,
                notes="Flat chunking + hash embeddings. Current system's retrieval."),
        Variant("hybrid_flat", lambda corpus: chunk_documents(corpus), use_hybrid=True,
                notes="Flat chunking + hash embeddings + RRF. The current baseline."),
        Variant("hybrid_structured", lambda corpus: chunk_documents_structured(corpus), use_hybrid=True,
                notes="Structure-aware chunking. Isolates the chunking contribution."),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval variants.")
    parser.add_argument("--variant", default="all", help="Run a single variant or 'all'.")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/latest.json"))
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    queries = resolve_queries(load_queries(args.queries), corpus)
    variants = default_variants()

    print(f"Corpus: {len(corpus)} documents | Queries: {len(queries)} | k={args.top_k}")
    report = run_benchmark(variants, corpus, queries, args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()