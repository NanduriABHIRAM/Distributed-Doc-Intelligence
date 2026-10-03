"""Build bounded, cited context for the answer-generation model."""

from __future__ import annotations


def build_context(chunks: list[dict], maximum_characters: int = 24_000) -> str:
    """Format retrieved chunks as untrusted reference material for an LLM.

    Source numbers are deliberately stable within a request: they are the
    citation identifiers the model must use in its answer.  The character
    limit prevents a large ``top_k`` from making a request unexpectedly large.
    """
    parts: list[str] = []
    remaining = maximum_characters
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        filename = metadata.get("filename", "unknown document")
        header = f"[Source {index}: {filename}]\n"
        if remaining <= len(header):
            break
        text = str(chunk.get("text", "")).strip()
        # Do not cut a UTF-8 byte stream; Python string slices are character
        # based.  An ellipsis tells the model that the evidence was truncated.
        available = remaining - len(header)
        if len(text) > available:
            text = text[: max(0, available - 1)].rstrip() + "…"
        parts.append(header + text)
        remaining -= len(parts[-1]) + 2
    return "\n\n".join(parts)
