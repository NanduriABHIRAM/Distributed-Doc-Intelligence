"""Split normalized documents into retrievable chunks."""


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 30) -> list[str]:
    """Split *text* into overlapping word chunks."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[start : start + chunk_size]) for start in range(0, len(words), step) if words[start : start + chunk_size]]


def chunk_documents(documents: list[dict], chunk_size: int = 180, overlap: int = 30) -> list[dict]:
    """Create chunks with source and document metadata preserved."""
    chunks = []
    for document in documents:
        for index, text in enumerate(chunk_text(document["clean_text"], chunk_size, overlap)):
            chunks.append(
                {
                    "chunk_id": f"{document['document_id']}-{index}",
                    "document_id": document["document_id"],
                    "source": document["source"],
                    "text": text,
                    "metadata": document["metadata"],
                }
            )
    return chunks
