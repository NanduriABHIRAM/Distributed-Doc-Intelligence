"""Extract text and page metadata from PDF documents.

``pypdf`` is used in ``layout`` extraction mode rather than its default. The
default mode drops the space between words when a PDF positions them with
explicit glyph coordinates instead of a space character, so prose arrives as
"theupdate()function" and "areknownas imperative". Those fused tokens are
invisible to term matching, so a document ingests cleanly, retrieves at the
right rank, and still fails to answer: the term simply is not in the text.

Layout mode preserves those inter-word gaps as runs of spaces instead of
discarding them. Collapsing the runs afterwards recovers the original words,
which makes the extracted text behave like normal prose for tokenization.

The trade-off is that layout mode keeps visual indentation. That is stripped by
the whitespace collapse, and any residual leading indent is trimmed per line so
headings do not inherit the margin of the page they were printed at.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# pdfplumber and some Tika paths emit this for undecodable glyphs.
_CID_MARKER = re.compile(r"\(cid:\d+\)")


def _repair_word_spacing(text: str) -> str:
    """Collapse layout-mode indentation back into single spaces.

    Layout mode emits a visual column layout rather than a reading order, so a
    line may begin with a wide indent and inter-word gaps become runs of spaces.
    Reducing runs to one space turns "If     the  update()" back into
    "If the update()" without altering the words themselves.
    """
    lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def strip_extraction_artifacts(text: str) -> str:
    """Remove extractor debris that would otherwise reach answers verbatim."""
    return _CID_MARKER.sub("", text)


def extract_pdf_text(pdf_path: str | Path) -> tuple[str, int]:
    """Return cleaned text and page count from a PDF using ``pypdf``."""
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF support requires `pip install pypdf`.") from error

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            # Layout mode is what preserves inter-word spacing; older pypdf
            # builds lack it, hence the fallback.
            raw = page.extract_text(extraction_mode="layout") or ""
        except (TypeError, ValueError, KeyError):
            raw = page.extract_text() or ""
        pages.append(_repair_word_spacing(strip_extraction_artifacts(raw)))
    return "\n".join(pages), len(reader.pages)