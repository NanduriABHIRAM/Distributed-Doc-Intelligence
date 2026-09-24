"""LLM provider abstraction point for a future hosted or local model."""


def generate_completion(prompt: str) -> str:
    """Raise a clear message until an LLM provider is configured."""
    raise RuntimeError("No LLM provider is configured. The prototype uses extractive answers instead.")
