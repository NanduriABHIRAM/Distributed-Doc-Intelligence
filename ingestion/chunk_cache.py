"""Persistent cache for extracted and chunked local documents.

The embedding cache avoids re-encoding chunks, but without this cache every
new terminal session still rereads every PDF before it can discover that the
embeddings are reusable.  This module makes normal question-answering startup
fast while invalidating safely whenever a supported source file changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from ingestion.document_loader import SUPPORTED_SUFFIXES

logger = logging.getLogger(__name__)


def corpus_fingerprint(source_path: str | Path) -> str:
    """Return a fingerprint of supported files beneath *source_path*."""
    source = Path(source_path)
    paths = [source] if source.is_file() else sorted(source.rglob("*"))
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        stat = path.stat()
        digest.update(str(path.relative_to(source) if source.is_dir() else path.name).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def load_chunk_cache(cache_path: str | Path, fingerprint: str) -> tuple[list[dict], list[dict]] | None:
    """Load cached document IDs and chunks if they match the raw corpus."""
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Ignoring unreadable chunk cache %s: %s", path, error)
        return None
    if payload.get("fingerprint") != fingerprint:
        logger.info("Chunk cache is stale; rebuilding it from source documents.")
        return None
    chunks = payload.get("chunks")
    documents = payload.get("documents")
    if not isinstance(chunks, list) or not isinstance(documents, list):
        logger.warning("Ignoring invalid chunk cache %s.", path)
        return None
    return documents, chunks


def save_chunk_cache(cache_path: str | Path, fingerprint: str, documents: list[dict], chunks: list[dict]) -> None:
    """Persist chunks before embeddings are added, keeping the cache compact."""
    path = Path(cache_path)
    serializable_chunks = [{key: value for key, value in chunk.items() if key != "embedding"} for chunk in chunks]
    payload = {
        "fingerprint": fingerprint,
        "documents": [{"document_id": document["document_id"]} for document in documents],
        "chunks": serializable_chunks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)
    logger.info("Cached %d chunks from %d documents at %s", len(chunks), len(documents), path)
