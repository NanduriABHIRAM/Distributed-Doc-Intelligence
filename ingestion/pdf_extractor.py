"""Extract text and page metadata from PDF documents."""

from pathlib import Path


def extract_pdf_text(pdf_path: str | Path) -> tuple[str, int]:
    """Return text and page count from a PDF using the optional ``pypdf`` package."""
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF support requires `pip install pypdf`.") from error

    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages), len(reader.pages)
