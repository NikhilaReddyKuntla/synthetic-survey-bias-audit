from __future__ import annotations

import base64
from pathlib import Path

from src.utils.pdf_utils import render_pdf_page_to_png_bytes

VISION_MODELS = {
    "groq_vision": ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"),
    "openai_vision": ("openai", "gpt-4.1-mini"),
}

VISUAL_SUMMARY_PROMPT = """
You are helping build a trusted RAG index for synthetic survey generation.
Analyze this PDF page image only for charts, graphs, figures, tables, or other visual evidence.

Return a concise summary that:
- identifies the visible visual type or says what kind of figure/table it appears to be
- summarizes the main trend or comparison
- notes important subgroup differences if visible
- avoids unsupported numbers unless they are clearly visible
- does not invent facts that are not visible on the page
""".strip()


def _image_url_from_pdf_page(pdf_path: Path, page_number: int, zoom: float = 1.5) -> str:
    image_bytes = render_pdf_page_to_png_bytes(pdf_path, page_number - 1, zoom=zoom)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{image_b64}"


def summarize_visual_page_with_groq(
    pdf_path: Path,
    page_number: int,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    image_zoom: float = 1.5,
) -> str:
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "Groq visual summaries require the Groq Python package. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    client = Groq()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISUAL_SUMMARY_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_url_from_pdf_page(pdf_path, page_number, zoom=image_zoom)},
                    },
                ],
            }
        ],
        temperature=0,
        max_completion_tokens=700,
    )

    summary = (completion.choices[0].message.content or "").strip()
    if not summary:
        raise RuntimeError(f"Groq vision model returned an empty summary for {pdf_path} page {page_number}.")

    return summary


def summarize_visual_page_with_openai(
    pdf_path: Path,
    page_number: int,
    model: str = "gpt-4.1-mini",
    detail: str = "low",
    image_zoom: float = 1.5,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI visual summaries require the OpenAI Python package. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": VISUAL_SUMMARY_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": _image_url_from_pdf_page(pdf_path, page_number, zoom=image_zoom),
                        "detail": detail,
                    },
                ],
            }
        ],
    )

    summary = getattr(response, "output_text", "").strip()
    if not summary:
        raise RuntimeError(f"OpenAI vision model returned an empty summary for {pdf_path} page {page_number}.")

    return summary


def summarize_visual_page(
    pdf_path: Path,
    page_number: int,
    provider: str = "groq",
    model: str | None = None,
    detail: str = "low",
    image_zoom: float = 1.5,
) -> str:
    if provider == "groq":
        return summarize_visual_page_with_groq(
            pdf_path=pdf_path,
            page_number=page_number,
            model=model or VISION_MODELS["groq_vision"][1],
            image_zoom=image_zoom,
        )

    if provider == "openai":
        return summarize_visual_page_with_openai(
            pdf_path=pdf_path,
            page_number=page_number,
            model=model or VISION_MODELS["openai_vision"][1],
            detail=detail,
            image_zoom=image_zoom,
        )

    raise ValueError("Unsupported vision provider. Expected 'groq' or 'openai'.")
