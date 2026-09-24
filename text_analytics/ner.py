"""Small rule-based named-entity extraction fallback."""

import re

ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


def extract_entities(text: str) -> list[dict]:
    """Return title-cased entity candidates without requiring an NLP model."""
    return [{"text": match.group(0), "label": "PROPER_NOUN"} for match in ENTITY_PATTERN.finditer(text)]
