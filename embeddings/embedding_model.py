"""Dependency-free hashed text embeddings for the prototype."""

import hashlib
import math

from text_analytics.preprocessing import tokenize


class HashEmbeddingModel:
    """Create deterministic normalized vectors from tokens.

    Replace this with Sentence Transformers for production semantic retrieval.
    """

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % self.dimensions] += -1.0 if value & 1 else 1.0
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector


def load_embedding_model() -> HashEmbeddingModel:
    """Load the dependency-free prototype embedding model."""
    return HashEmbeddingModel()
