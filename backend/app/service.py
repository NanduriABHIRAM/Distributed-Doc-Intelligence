"""Shared, cached pipeline instance used by FastAPI routes."""

from functools import lru_cache
from pathlib import Path

from pipeline import DocumentIntelligencePipeline


@lru_cache
def get_pipeline() -> DocumentIntelligencePipeline:
    """Build the document indexes once per API process."""
    data_directory = Path(__file__).resolve().parents[2] / "data" / "raw"
    return DocumentIntelligencePipeline(str(data_directory)).build()


def public_results(results: dict) -> dict:
    """Remove internal embedding vectors from API responses."""
    return {name: [{key: value for key, value in item.items() if key != "embedding"} for item in items] for name, items in results.items()}
