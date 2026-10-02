"""Structure-aware document chunking.

Chunks are split on document structure first — headings, slide markers, and
bullet blocks — and only fall back to size-based splitting inside sections that
are too large. Section titles are carried into chunk metadata so a retrieved
passage keeps the context of where it came from, which matters most for slide
decks where an individual bullet carries no meaning on its own.

The output schema deliberately matches `chunking.chunker.chunk_documents` so
results remain interchangeable with the Scala Spark batch job.
"""

from __future__ import annotations

import re

from text_analytics.keyword_extraction import extract_keywords

# Structural cues that reliably mark a boundary in lecture material.
# Kept case-insensitive, and deliberately restricted to single lines.
_HEADING_PATTERN = re.compile(
    r"^[ \t]*(?:"
    r"(?:unit|chapter|lecture|topic|section|module|part)\s+[0-9ivxlc]+"   # Unit 3 / Lecture 4
    r"|[0-9]+(?:\.[0-9]+){1,3}\.?[ \t]+[A-Z][^\n]{0,80}"                  # 3.2.1 Operating Systems
    r")[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Slide markers usually carry a trailing title ("Slide 2: Types of OS"), so they
# are matched separately rather than being forced to end at the colon.
_SLIDE_HEADING_PATTERN = re.compile(
    r"^[ \t]*slide\s+[0-9]+[ \t]*[:.\-]?[ \t]*[^\n]{0,80}$",
    re.IGNORECASE | re.MULTILINE,
)

# ALL-CAPS section titles must be matched case-sensitively. Folding this into
# the case-insensitive pattern above would make every lowercase line a heading.
_ALLCAPS_PATTERN = re.compile(
    r"^[ \t]*[A-Z][A-Z0-9 &/,'-]{4,60}[ \t]*:?[ \t]*$",
    re.MULTILINE,
)

# A "heading" must not simply be a long run of body prose.
_PROSE_HINT = re.compile(r"[a-z]")

_SLIDE_MARKER = re.compile(r"\bSlide\s+(\d+)\b", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•‣▪●·o]|\(?[a-z0-9]{1,2}[.)]|\d+[.)])\s+", re.IGNORECASE)


def _heading_spans(text: str) -> list[re.Match]:
    """Collect genuine heading matches from both patterns, ordered by position."""
    spans = [match for match in _HEADING_PATTERN.finditer(text)]
    spans.extend(match for match in _SLIDE_HEADING_PATTERN.finditer(text))
    spans.extend(match for match in _ALLCAPS_PATTERN.finditer(text) if not _PROSE_HINT.search(match.group(0)))
    spans.sort(key=lambda match: match.start())
    return spans


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split *text* into (section_title, section_body) pairs.

    Text before the first recognised heading is treated as a preamble section.
    If no headings are found the whole document becomes one section, so the
    caller transparently falls back to size-based splitting.
    """
    matches = _heading_spans(text)
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for index, match in enumerate(matches):
        title = match.group(0).strip().strip(":")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((title, body))
    return sections


def _split_oversized(body: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a section that exceeds *chunk_size* words.

    Bullet blocks are kept together where possible so an enumerated list is not
    cut mid-way; anything still too long is split on word count.
    """
    if len(body.split()) <= chunk_size:
        return [body]

    blocks = [block for block in re.split(r"\n\s*\n", body) if block.strip()]
    if len(blocks) <= 1:
        blocks = [block for block in body.splitlines() if block.strip()] or [body]

    grouped: list[str] = []
    current: list[str] = []
    current_length = 0
    for block in blocks:
        block_words = len(block.split())
        # A single oversized block is flushed on its own rather than merged.
        if current and current_length + block_words > chunk_size:
            grouped.append("\n".join(current))
            current, current_length = [], 0
        current.append(block)
        current_length += block_words
    if current:
        grouped.append("\n".join(current))

    pieces: list[str] = []
    for group in grouped:
        words = group.split()
        if len(words) <= chunk_size:
            pieces.append(group)
            continue
        step = max(chunk_size - overlap, 1)
        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            if window:
                pieces.append(" ".join(window))
    return pieces


def _section_title_for(text: str) -> str:
    """Best-effort short label for a chunk, used when no heading was found."""
    slide = _SLIDE_MARKER.search(text)
    if slide:
        return f"Slide {slide.group(1)}"
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    first_line = _BULLET.sub("", first_line).strip()
    if 3 <= len(first_line) <= 80:
        return first_line
    return ""


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 30) -> list[dict]:
    """Chunk *text* on structure, returning dicts with text and section title."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")

    sections = _split_into_sections(text)
    chunks: list[dict] = []
    for title, body in sections:
        if not body.strip():
            continue
        for index, piece in enumerate(_split_oversized(body, chunk_size, overlap)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "text": piece,
                    "section_title": title or _section_title_for(piece),
                    "section_index": len(chunks),
                }
            )
    return chunks or [{"text": text, "section_title": "", "section_index": 0}]


def chunk_documents_structured(
    documents: list[dict], chunk_size: int = 180, overlap: int = 30
) -> list[dict]:
    """Create structure-aware chunks with source and document metadata preserved."""
    chunks: list[dict] = []
    for document in documents:
        for index, piece in enumerate(chunk_text(document["clean_text"], chunk_size, overlap)):
            metadata = dict(document["metadata"])
            metadata["section_title"] = piece["section_title"]
            chunks.append(
                {
                    "chunk_id": f"{document['document_id']}-{index}",
                    "document_id": document["document_id"],
                    "source": document["source"],
                    "text": piece["text"],
                    "metadata": metadata,
                    "keywords": extract_keywords(piece["text"], top_k=12),
                }
            )
    return chunks