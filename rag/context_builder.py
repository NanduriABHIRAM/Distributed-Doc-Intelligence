"""Build LLM context from retrieved chunks."""


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a cited, LLM-ready context string."""
    return "\n\n".join(f"[Source {index}] {chunk['metadata']['filename']}\n{chunk['text']}" for index, chunk in enumerate(chunks, start=1))
