"""Hybrid search endpoints."""

from fastapi import APIRouter

from backend.app.schemas.search import SearchRequest
from backend.app.service import get_pipeline, public_results

router = APIRouter()


@router.post("/")
def search_documents(request: SearchRequest):
    """Run BM25, vector, and fused hybrid search."""
    return {"query": request.query, "results": public_results(get_pipeline().search(request.query, request.top_k))}
