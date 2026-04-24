from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from src.utils.pdf_utils import extract_pdf_text
from src.utils.text_utils import clean_text

SUPPORTED_DOCUMENT_EXTENSIONS = (".pdf", ".txt", ".md", ".docx", ".json")


def supported_document(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS


def extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        with archive.open("word/document.xml") as handle:
            root = ElementTree.parse(handle).getroot()

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []

    for paragraph in root.findall(".//w:p", namespace):
        text_runs = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        paragraph_text = "".join(text_runs).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    return "\n".join(paragraphs)


def extract_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix == ".docx":
        return extract_docx_text(path)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return json.dumps(payload, ensure_ascii=False, indent=2)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(
        f"Unsupported document type for {path}. "
        f"Supported extensions: {', '.join(SUPPORTED_DOCUMENT_EXTENSIONS)}"
    )


def extract_clean_document_text(path: Path) -> str:
    return clean_text(extract_document_text(path))
