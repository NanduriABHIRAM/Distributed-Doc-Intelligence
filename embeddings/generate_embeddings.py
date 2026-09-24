"""Generate embeddings for document chunks."""

from embeddings.embedding_model import HashEmbeddingModel, load_embedding_model


def generate_embeddings(chunks: list[dict], model: HashEmbeddingModel | None = None) -> list[dict]:
    """Attach a vector embedding to each chunk and return the chunks."""
    model = model or load_embedding_model()
    for chunk in chunks:
        chunk["embedding"] = model.encode(chunk["text"])
    return chunks
