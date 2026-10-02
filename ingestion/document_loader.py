"""Load supported documents from a file or directory."""

from pathlib import Path

from ingestion.office_extractor import extract_docx_text, extract_presentation_text
from ingestion.pdf_extractor import extract_pdf_text

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".pptx", ".ppsx"}


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
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text, page_count = extract_pdf_text(path)
            unit_count, unit_name = page_count, "page_count"
        elif suffix in {".pptx", ".ppsx"}:
            text, slide_count = extract_presentation_text(path)
            unit_count, unit_name = slide_count, "slide_count"
        elif suffix == ".docx":
            text, paragraph_count = extract_docx_text(path)
            unit_count, unit_name = paragraph_count, "paragraph_count"
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            unit_count, unit_name = None, None
        if text.strip():
            metadata = {"filename": path.name}
            if unit_name:
                metadata[unit_name] = unit_count
            documents.append(
                {
                    "document_id": path.stem,
                    "source": str(path),
                    "text": text,
                    "metadata": metadata,
                }
            )
    return documents
