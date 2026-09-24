"""Text normalization and tokenization utilities."""

import re

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def preprocess_text(text: str) -> str:
    """Normalize whitespace while retaining case for readable answers."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Return lowercase word tokens used for retrieval and analytics."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]
