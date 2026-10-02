"""Generate embeddings for document chunks, with an on-disk cache.

Encoding the full corpus is the slowest step at startup -- roughly 13,000
chunks through a MiniLM encoder -- and it is repeated on every launch even
though the encoder is frozen and the corpus rarely changes between runs. The
cache makes repeat launches near-instant while staying correct: the key folds
in the model identity and a digest of the chunk texts, so an edited document,
a different chunker, or a different model all invalidate it automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path

from embeddings.embedding_cache import (
    DEFAULT_CACHE,
    describe,
    load_cached_vectors,
    save_cached_vectors,
)
from embeddings.sentence_encoder import load_embedding_model

logger = logging.getLogger(__name__)


def generate_embeddings(chunks: list[dict], model=None, cache_path: Path | None = DEFAULT_CACHE, use_cache: bool = True) -> list[dict]:
    """Attach an embedding vector to each chunk.

    Falls back to uncached encoding whenever anything about the cache is
    missing, stale, or unwritable, so caching can only make a run faster and
    never change its result.
    """
    model = model or load_embedding_model()
    if not chunks:
        return chunks

    model_name = describe(model)
    dimensions = int(getattr(model, "dimensions", 0) or 0)

    if use_cache and cache_path is not None and dimensions:
        cached = load_cached_vectors(Path(cache_path), chunks, model_name, dimensions)
        if cached is not None:
            for chunk in chunks:
                chunk["embedding"] = cached[chunk["chunk_id"]]
            logger.info("Reused %d cached embeddings from %s", len(chunks), cache_path)
            return chunks

    texts = [str(chunk["text"]) for chunk in chunks]
    if hasattr(model, "encode_batch"):
        vectors = model.encode_batch(texts)
    else:
        vectors = [model.encode(text) for text in texts]
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk["embedding"] = vector

    if use_cache and cache_path is not None and dimensions:
        try:
            save_cached_vectors(Path(cache_path), chunks, model_name, dimensions)
        except OSError as error:
            # A read-only or full disk must not fail the run; the embeddings are
            # already computed and the pipeline can use them.
            logger.warning("Could not write embedding cache %s: %s", cache_path, error)
    return chunks
