"""Schemas for retrieval and answer responses."""

from pydantic import BaseModel


class SourceReference(BaseModel):
    """Reference to a document chunk used as evidence."""

    document_name: str
    page: int | None = None
    score: float | None = None
