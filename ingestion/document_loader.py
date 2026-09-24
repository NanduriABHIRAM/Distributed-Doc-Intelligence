"""Load supported documents from a file or directory."""

from pathlib import Path

from ingestion.pdf_extractor import extract_pdf_text

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def load_documents(source_path: str | Path) -> list[dict]:
    """Return text documents and source metadata found below *source_path*."""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Document source does not exist: {source}")

    paths = [source] if source.is_file() else sorted(source.rglob("*"))
    documents = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.suffix.lower() == ".pdf":
            text, page_count = extract_pdf_text(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            page_count = None
        if text.strip():
            documents.append(
                {
                    "document_id": path.stem,
                    "source": str(path),
                    "text": text,
                    "metadata": {"filename": path.name, "page_count": page_count},
                }
            )
    return documents
