"""Retrieval-augmented question-answering endpoints."""

from fastapi import APIRouter

from backend.app.schemas.search import SearchRequest
from backend.app.service import get_pipeline, public_results

router = APIRouter()


@router.post("/")
def answer_question(request: SearchRequest):
    """Answer a question with source-grounded extractive RAG output."""
    response = get_pipeline().answer(request.query, request.top_k)
    response["retrieval"] = public_results(response["retrieval"])
    return response
