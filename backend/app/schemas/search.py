"""Schemas for search and question-answering requests."""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """A natural-language query submitted to the system."""

    query: str = Field(min_length=1, description="Question or search query")
    top_k: int = Field(default=5, ge=1, le=50)
