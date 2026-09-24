"""Coordinate document loading and normalization."""

from ingestion.document_loader import load_documents
from text_analytics.preprocessing import preprocess_text


def run_ingestion(source_path: str) -> list[dict]:
    """Load documents and attach cleaned text ready for chunking."""
    documents = load_documents(source_path)
    for document in documents:
        document["clean_text"] = preprocess_text(document["text"])
    return documents
