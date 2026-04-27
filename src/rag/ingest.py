from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.rag.vision import summarize_visual_page
from src.utils.helpers import data_dir, rag_docs_path, read_jsonl, write_jsonl
from src.utils.pdf_utils import extract_pdf_pages_text, page_has_visual_content
from src.utils.text_utils import chunk_text, clean_text, extract_year

ALLOWED_DOMAINS = ("ecommerce", "finance", "healthcare")
SUPPORTED_SOURCE_EXTENSIONS = (".pdf", ".txt", ".md")


def infer_doc_type(file_name: str) -> str:
    lowered = file_name.lower()
    if "synthetic" in lowered or "product_x" in lowered or "product-x" in lowered:
        return "synthetic_project_context"
    if "data-book" in lowered:
        return "data_book"
    if "insurance" in lowered:
        return "health_stats"
    if "quarterly" in lowered or "quaterly" in lowered:
        return "quarterly_report"
    if "report" in lowered:
        return "report"
    return "document"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "document"


def infer_domain(pdf_path: Path, domain: str | None = None) -> str:
    if domain:
        if domain not in ALLOWED_DOMAINS:
            raise ValueError(f"Unsupported domain '{domain}'. Expected one of: {', '.join(ALLOWED_DOMAINS)}")
        return domain

    parent_name = pdf_path.parent.name
    if parent_name in ALLOWED_DOMAINS:
        return parent_name

    raise ValueError(
        f"Could not infer domain for {pdf_path}. "
        f"Pass --domain with one of: {', '.join(ALLOWED_DOMAINS)}"
    )


def discover_source_pdfs(input_paths: list[Path] | None = None, domain: str | None = None) -> list[tuple[Path, str]]:
    if input_paths:
        discovered: list[tuple[Path, str]] = []
        for input_path in input_paths:
            paths = (
                sorted(path for path in input_path.rglob("*") if path.suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS)
                if input_path.is_dir()
                else [input_path]
            )
            for pdf_path in paths:
                if pdf_path.suffix.lower() not in SUPPORTED_SOURCE_EXTENSIONS:
                    continue
                discovered.append((pdf_path, infer_domain(pdf_path, domain=domain)))
        return discovered

    raw_sources_dir = data_dir() / "raw_sources"
    pdfs: list[tuple[Path, str]] = []

    for allowed_domain in ALLOWED_DOMAINS:
        domain_dir = raw_sources_dir / allowed_domain
        if not domain_dir.exists():
            continue
        pdfs.extend(
            (source_path, allowed_domain)
            for source_path in sorted(domain_dir.rglob("*"))
            if source_path.suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS
        )

    return pdfs


def build_text_file_chunks(
    source_path: Path,
    domain: str,
    chunk_size: int = 450,
    chunk_overlap: int = 80,
) -> list[dict]:
    file_name = source_path.name
    year = extract_year(file_name)
    doc_type = infer_doc_type(file_name)
    file_stem = slugify(source_path.stem)

    try:
        raw_text = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = source_path.read_text(encoding="utf-8", errors="ignore")

    cleaned_text = clean_text(raw_text)
    if not cleaned_text:
        return []

    return [
        {
            "chunk_id": f"{domain}_{file_stem}_text_{chunk_index:03d}",
            "chunk_type": "text",
            "domain": domain,
            "source_file": file_name,
            "doc_type": doc_type,
            "year": year,
            "text": chunk,
        }
        for chunk_index, chunk in enumerate(
            chunk_text(cleaned_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            start=1,
        )
    ]


def build_pdf_chunks(
    pdf_path: Path,
    domain: str,
    chunk_size: int = 450,
    chunk_overlap: int = 80,
    include_visual_summaries: bool = False,
    vision_provider: str = "groq",
    vision_model: str | None = None,
    vision_detail: str = "low",
    vision_image_zoom: float = 1.5,
    local_vision_endpoint: str = "http://localhost:11434/api/chat",
) -> list[dict]:
    if pdf_path.suffix.lower() in {".txt", ".md"}:
        return build_text_file_chunks(
            source_path=pdf_path,
            domain=domain,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    chunks: list[dict] = []
    file_name = pdf_path.name
    year = extract_year(file_name)
    doc_type = infer_doc_type(file_name)
    file_stem = slugify(pdf_path.stem)

    try:
        page_texts = extract_pdf_pages_text(pdf_path)
    except Exception as exc:
        print(f"Skipping unreadable PDF: {pdf_path} ({exc})")
        return chunks

    raw_text = "\n".join(page_texts)
    cleaned_text = clean_text(raw_text)
    if cleaned_text:
        file_chunks = chunk_text(cleaned_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk_index, chunk in enumerate(file_chunks, start=1):
            chunks.append(
                {
                    "chunk_id": f"{domain}_{file_stem}_text_{chunk_index:03d}",
                    "chunk_type": "text",
                    "domain": domain,
                    "source_file": file_name,
                    "doc_type": doc_type,
                    "year": year,
                    "text": chunk,
                }
            )
    else:
        print(f"No extractable normal text found in PDF: {pdf_path}")

    if include_visual_summaries:
        for page_index, page_text in enumerate(page_texts):
            if not page_has_visual_content(pdf_path, page_index, page_text=page_text):
                continue

            page_number = page_index + 1
            try:
                summary = summarize_visual_page(
                    pdf_path=pdf_path,
                    page_number=page_number,
                    provider=vision_provider,
                    model=vision_model,
                    detail=vision_detail,
                    image_zoom=vision_image_zoom,
                    local_endpoint=local_vision_endpoint,
                )
            except Exception as exc:
                print(f"Skipping visual summary for {pdf_path} page {page_number} ({exc})")
                continue

            chunks.append(
                {
                    "chunk_id": f"{domain}_{file_stem}_visual_p{page_number:03d}",
                    "chunk_type": "visual_summary",
                    "domain": domain,
                    "source_file": file_name,
                    "doc_type": doc_type,
                    "year": year,
                    "page_number": page_number,
                    "text": summary,
                }
            )

    return chunks


def build_chunks(
    chunk_size: int = 450,
    chunk_overlap: int = 80,
    input_paths: list[Path] | None = None,
    domain: str | None = None,
    include_visual_summaries: bool = False,
    vision_provider: str = "groq",
    vision_model: str | None = None,
    vision_detail: str = "low",
    vision_image_zoom: float = 1.5,
    local_vision_endpoint: str = "http://localhost:11434/api/chat",
) -> list[dict]:
    chunks: list[dict] = []

    for pdf_path, pdf_domain in discover_source_pdfs(input_paths=input_paths, domain=domain):
        chunks.extend(
            build_pdf_chunks(
                pdf_path=pdf_path,
                domain=pdf_domain,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                include_visual_summaries=include_visual_summaries,
                vision_provider=vision_provider,
                vision_model=vision_model,
                vision_detail=vision_detail,
                vision_image_zoom=vision_image_zoom,
                local_vision_endpoint=local_vision_endpoint,
            )
        )

    return chunks


def merge_chunks(existing_chunks: list[dict], new_chunks: list[dict]) -> list[dict]:
    merged = {chunk["chunk_id"]: chunk for chunk in existing_chunks}
    for chunk in new_chunks:
        merged[chunk["chunk_id"]] = chunk
    return list(merged.values())


def ingest_documents(
    chunk_size: int = 450,
    chunk_overlap: int = 80,
    output_path: Path | None = None,
    input_paths: list[Path] | None = None,
    domain: str | None = None,
    append: bool = False,
    include_visual_summaries: bool = False,
    vision_provider: str = "groq",
    vision_model: str | None = None,
    vision_detail: str = "low",
    vision_image_zoom: float = 1.5,
    local_vision_endpoint: str = "http://localhost:11434/api/chat",
) -> Path:
    output_path = output_path or rag_docs_path()
    chunks = build_chunks(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        input_paths=input_paths,
        domain=domain,
        include_visual_summaries=include_visual_summaries,
        vision_provider=vision_provider,
        vision_model=vision_model,
        vision_detail=vision_detail,
        vision_image_zoom=vision_image_zoom,
        local_vision_endpoint=local_vision_endpoint,
    )
    if append and output_path.exists():
        chunks = merge_chunks(read_jsonl(output_path), chunks)
    write_jsonl(output_path, chunks)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JSONL chunks from curated RAG PDFs.")
    parser.add_argument("--chunk-size", type=int, default=450, help="Chunk size in words.")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="Chunk overlap in words.")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        dest="input_paths",
        help="PDF file or directory to ingest. Can be provided multiple times.",
    )
    parser.add_argument("--domain", choices=ALLOWED_DOMAINS, help="Domain for --input documents outside domain folders.")
    parser.add_argument("--append", action="store_true", help="Append/upsert chunks into an existing chunk JSONL.")
    parser.add_argument(
        "--visual-summaries",
        action="store_true",
        help="Generate separate vision summaries for pages that appear to contain charts, figures, or graphs.",
    )
    parser.add_argument(
        "--vision-provider",
        default="groq",
        choices=("groq", "local", "openai"),
        help="Vision LLM provider.",
    )
    parser.add_argument("--vision-model", default=None, help="Vision-capable LLM model.")
    parser.add_argument("--vision-detail", default="low", choices=("low", "high", "auto"), help="Image detail level.")
    parser.add_argument("--vision-image-zoom", type=float, default=1.5, help="PDF page render zoom for vision input.")
    parser.add_argument(
        "--local-vision-endpoint",
        default="http://localhost:11434/api/chat",
        help="Local Ollama-compatible chat endpoint for --vision-provider local.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=rag_docs_path(),
        help="Where to write the processed chunk JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = ingest_documents(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_path=args.output,
        input_paths=args.input_paths,
        domain=args.domain,
        append=args.append,
        include_visual_summaries=args.visual_summaries,
        vision_provider=args.vision_provider,
        vision_model=args.vision_model,
        vision_detail=args.vision_detail,
        vision_image_zoom=args.vision_image_zoom,
        local_vision_endpoint=args.local_vision_endpoint,
    )
    print(f"Wrote RAG chunks to {output_path}")


if __name__ == "__main__":
    main()
