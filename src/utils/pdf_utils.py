from __future__ import annotations

from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - fallback for older environments
    from PyPDF2 import PdfReader  # type: ignore[no-redef]


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    return "\n".join(pages)


def extract_pdf_pages_text(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def page_has_visual_content(pdf_path: Path, page_index: int, page_text: str = "") -> bool:
    lowered_text = page_text.lower()
    visual_markers = ("figure", "fig.", "chart", "graph", "exhibit")
    has_visual_marker = any(marker in lowered_text for marker in visual_markers)

    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        return has_visual_marker

    document = fitz.open(pdf_path)
    try:
        page = document.load_page(page_index)
        has_images = bool(page.get_images(full=True))
        has_many_drawings = len(page.get_drawings()) >= 8
        return has_images or has_many_drawings or has_visual_marker
    finally:
        document.close()


def render_pdf_page_to_png_bytes(pdf_path: Path, page_index: int, zoom: float = 2.0) -> bytes:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Visual summaries require PyMuPDF. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    document = fitz.open(pdf_path)
    try:
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pixmap.tobytes("png")
    finally:
        document.close()
