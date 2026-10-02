"""Extract readable text from Office Open XML documents without extra packages."""

from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree


OFFICE_DOCUMENT_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DRAWING_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _read_xml_text(archive: ZipFile, member: str, namespace: str) -> str:
    """Return text runs from an XML document stored in an Office archive."""
    root = ElementTree.fromstring(archive.read(member))
    return " ".join(node.text for node in root.iter(f"{{{namespace}}}t") if node.text)


def extract_presentation_text(presentation_path: str | Path) -> tuple[str, int]:
    """Return slide text and slide count from a PPTX or PPSX presentation."""
    with ZipFile(presentation_path) as archive:
        slide_files = sorted(
            (name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")),
            key=lambda name: int(name.rsplit("slide", 1)[1].split(".", 1)[0]),
        )
        slides = []
        for index, slide_file in enumerate(slide_files, 1):
            slide_text = _read_xml_text(archive, slide_file, DRAWING_NAMESPACE)
            if slide_text:
                slides.append(f"Slide {index}: {slide_text}")
    return "\n".join(slides), len(slides)


def extract_docx_text(docx_path: str | Path) -> tuple[str, int]:
    """Return text from a DOCX document and its paragraph count."""
    with ZipFile(docx_path) as archive:
        text = _read_xml_text(archive, "word/document.xml", OFFICE_DOCUMENT_NAMESPACE)
    return text, len([paragraph for paragraph in text.splitlines() if paragraph.strip()])
