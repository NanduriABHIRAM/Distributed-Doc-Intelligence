"""Sentence-transformer embedding model with a dependency-free fallback.

The prototype previously used a hashed bag-of-words, which has no notion of
meaning: a query and a passage using entirely different words score zero even
when they describe the same thing. This module provides a real bi-encoder
behind the same `encode()` interface, so retrieval code is unchanged.

If sentence-transformers or the model weights are unavailable, loading falls
back to the hashed model so the pipeline still runs — degraded, but not broken.

Model choice: paraphrase-multilingual-MiniLM-L12-v2 by default. The corpus is
Indian university lecture material containing code-switched English, LaTeX
maths notation, and heavy abbreviation; a multilingual encoder handles that
better than an English-only MiniLM at the same size.

Usage:
    pip install sentence-transformers
    DDI_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
"""

from __future__ import annotations

import logging
import os

from embeddings.embedding_model import HashEmbeddingModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("DDI_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


def _tokenizer_safe_text(text: object) -> str:
    """Return text with malformed Unicode replaced for the Rust tokenizer.

    PDF extraction can produce unpaired Unicode surrogate characters. Python's
    string functions accept them, but Hugging Face's fast tokenizer rejects
    them with ``TextEncodeInput``. Replacing only invalid code points preserves
    all normal document text while allowing the batch to continue.
    """
    return str(text).encode("utf-8", errors="replace").decode("utf-8")


class SentenceEncoder:
    """Encode text with a sentence-transformers bi-encoder.

    Exposes the same `encode(text) -> list[float]` contract as
    HashEmbeddingModel, returning L2-normalized vectors so cosine similarity is
    a plain dot product.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu", batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.dimensions = 0
        self._model = SentenceTransformer(model_name, device=device)
        self.dimensions = int(self._model.get_embedding_dimension())

    def encode(self, text: str) -> list[float]:
        # Sentence Transformers accepts a single string, but explicitly batching it
        # avoids tokenizer input-shape issues with some recent dependency versions.
        vector = self._model.encode(
            [_tokenizer_safe_text(text)],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return [float(value) for value in vector]

    def encode_batch(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """Encode many texts at once; substantially faster than per-item calls."""
        vectors = self._model.encode(
            [_tokenizer_safe_text(text) for text in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return [[float(value) for value in vector] for vector in vectors]


def load_embedding_model(prefer_semantic: bool = True):
    """Load the best available embedding model.

    Attempts a real sentence-transformer first and falls back to the hashed
    model, logging which path was taken so results are never silently mixed.
    """
    if prefer_semantic:
        try:
            model = SentenceEncoder()
            logger.info("Loaded semantic embedding model %s (%d dims)", model.model_name, model.dimensions)
            return model
        except Exception as error:  # noqa: BLE001 - fallback is the point
            logger.warning(
                "Falling back to hashed embeddings: %s. "
                "Install with `pip install sentence-transformers` for semantic retrieval.",
                error,
            )
    return HashEmbeddingModel()


def model_info() -> dict:
    """Describe the active embedding model, for run metadata and reports."""
    model = load_embedding_model()
    if isinstance(model, SentenceEncoder):
        return {"type": "sentence-transformer", "model": model.model_name, "dimensions": model.dimensions}
    return {"type": "hash-fallback", "model": "HashEmbeddingModel", "dimensions": model.dimensions}
