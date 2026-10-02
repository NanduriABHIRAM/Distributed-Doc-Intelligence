"""Coordinate document loading and normalization."""

from ingestion.document_loader import load_documents
from text_analytics.keyword_extraction import extract_keywords
from text_analytics.ner import extract_entities
from text_analytics.preprocessing import preprocess_text


def run_ingestion(source_path: str) -> list[dict]:
    """Load documents and attach cleaned text ready for chunking."""
    documents = load_documents(source_path)
    for document in documents:
        document["clean_text"] = preprocess_text(document["text"])
        document["metadata"]["keywords"] = extract_keywords(document["clean_text"])
        document["metadata"]["entities"] = extract_entities(document["text"])[:25]
    return documents
