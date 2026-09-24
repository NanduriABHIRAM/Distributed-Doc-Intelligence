"""Document management endpoints."""

from fastapi import APIRouter

from backend.app.service import get_pipeline

router = APIRouter()


@router.get("/")
def list_documents():
    """List documents loaded from the raw-data directory."""
    pipeline = get_pipeline()
    return {"count": len(pipeline.documents), "documents": [{"id": item["document_id"], "filename": item["metadata"]["filename"]} for item in pipeline.documents]}
