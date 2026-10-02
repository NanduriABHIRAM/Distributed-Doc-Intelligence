"""Disk cache for chunk embeddings.

Re-embedding the whole corpus on every launch dominated startup: ~13,000
chunks through a MiniLM takes several minutes, all of it repeated to answer
questions that did not change. Since the encoder is frozen -- nothing is
fine-tuned anywhere in this project -- a chunk's vector depends only on the
chunk text and the model identity, and is therefore safe to reuse.

The cache key folds in the model name, the vector width, and a digest of every
chunk id and text. Editing a document, changing the chunker, or switching
models therefore invalidates it automatically instead of silently serving
vectors that no longer match the corpus.

Usage:
    python -m embeddings.embedding_cache --data-dir data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE = Path(os.environ.get("DDI_EMBEDDING_CACHE", "data/processed/embeddings.json"))


def fingerprint(chunks: list[dict], model_name: str, dimensions: int) -> str:
    """Stable digest of the corpus and the model that will embed it."""
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(str(dimensions).encode("utf-8"))
    for chunk in chunks:
        digest.update(str(chunk.get("chunk_id", "")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(chunk.get("text", "")).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def load_cached_vectors(cache_path: Path, chunks: list[dict], model_name: str, dimensions: int) -> dict[str, list[float]] | None:
    """Return ``chunk_id -> vector`` if the cache matches this corpus exactly."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as error:
        logger.warning("Ignoring unreadable embedding cache %s: %s", cache_path, error)
        return None

    expected = fingerprint(chunks, model_name, dimensions)
    if payload.get("fingerprint") != expected:
        logger.info("Embedding cache is stale (corpus or model changed); rebuilding.")
        return None
    vectors = payload.get("vectors", {})
    missing = [chunk["chunk_id"] for chunk in chunks if chunk["chunk_id"] not in vectors]
    if missing:
        logger.warning("Embedding cache missing %d chunk(s); rebuilding.", len(missing))
        return None
    return vectors


def save_cached_vectors(cache_path: Path, chunks: list[dict], model_name: str, dimensions: int) -> None:
    """Persist vectors for every chunk currently carrying an embedding."""
    vectors = {
        chunk["chunk_id"]: chunk["embedding"]
        for chunk in chunks
        if "embedding" in chunk
    }
    if not vectors:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "dimensions": dimensions,
        "fingerprint": fingerprint(chunks, model_name, dimensions),
        "vectors": vectors,
    }
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    logger.info("Cached %d embeddings to %s", len(vectors), cache_path)


def describe(model) -> str:
    """A short human-readable name for the active encoder."""
    return getattr(model, "model_name", None) or type(model).__name__
