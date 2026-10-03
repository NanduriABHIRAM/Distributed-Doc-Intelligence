"""Extract readable text from Office Open XML documents without extra packages."""

from pathlib import Path
import re
from zipfile import ZipFile
from xml.etree import ElementTree


OFFICE_DOCUMENT_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DRAWING_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _read_xml_text(archive: ZipFile, member: str, namespace: str) -> str:
    """Return text runs from an XML document stored in an Office archive."""
    root = ElementTree.fromstring(archive.read(member))
    return " ".join(node.text for node in root.iter(f"{{{namespace}}}t") if node.text)


def _slide_paragraphs(archive: ZipFile, slide_file: str) -> list[str]:
    """Return each text shape on a slide as its own line.

    A slide is a set of independent shapes -- a title, bullets, a diagram
    label -- and each carries its own meaning. Joining them into one line
    destroys the boundaries the chunker relies on: heading patterns are
    anchored to line starts, so with a whole slide collapsed onto one line the
    chunker could only ever see the *first* "Slide N:" marker. Every later
    slide ran on inside the previous one's body, producing chunks titled
    "Slide 4" whose text begins "Slide 5: Process States", and answers that
    spliced two unrelated slides together.
    """
    root = ElementTree.fromstring(archive.read(slide_file))
    tag = f"{{{DRAWING_NAMESPACE}}}p"
    paragraphs: list[str] = []
    for paragraph in root.iter(tag):
        text = "".join(
            node.text or ""
            for node in paragraph.iter(f"{{{DRAWING_NAMESPACE}}}t")
        ).strip()
        if text:
            paragraphs.append(re.sub(r"\s+", " ", text))
    return paragraphs


def extract_presentation_text(presentation_path: str | Path) -> tuple[str, int]:
    """Return slide text and slide count from a PPTX or PPSX presentation."""
    with ZipFile(presentation_path) as archive:
        slide_files = sorted(
            (name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")),
            key=lambda name: int(name.rsplit("slide", 1)[1].split(".", 1)[0]),
        )
        slides = []
        for index, slide_file in enumerate(slide_files, 1):
            paragraphs = _slide_paragraphs(archive, slide_file)
            if not paragraphs:
                continue
            # The marker occupies its own line so the chunker sees a heading
            # boundary at the true start of this slide's content.
            slides.append(f"Slide {index}")
            slides.extend(paragraphs)
    return "\n".join(slides), len(slide_files)


def extract_docx_text(docx_path: str | Path) -> tuple[str, int]:
    """Return text from a DOCX document and its paragraph count."""
    with ZipFile(docx_path) as archive:
        text = _read_xml_text(archive, "word/document.xml", OFFICE_DOCUMENT_NAMESPACE)
    return text, len([paragraph for paragraph in text.splitlines() if paragraph.strip()])
