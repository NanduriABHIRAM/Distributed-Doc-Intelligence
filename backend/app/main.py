"""FastAPI application entrypoint."""

from fastapi import FastAPI

from backend.app.api import documents, qa, search

app = FastAPI(title="Distributed Document Intelligence API")
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(qa.router, prefix="/qa", tags=["question-answering"])
