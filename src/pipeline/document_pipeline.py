from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

import faiss
import numpy as np

from src.attacks.poison_utils import build_adversarial_templates
from src.generation.generate_responses import call_generation_model
from src.rag.embed import DEFAULT_MODEL, load_embedding_model
from src.rag.retrieve import format_retrieved_context
from src.utils.helpers import ensure_parent_dir, read_json, vector_store_dir, write_json
from src.utils.pdf_utils import extract_pdf_pages_text
from src.utils.prompt_templates import build_survey_response_prompt
from src.utils.text_utils import chunk_text, clean_text, extract_year

ALLOWED_DOMAINS = ("ecommerce", "finance", "healthcare")
DEFAULT_ADVERSARIAL_THRESHOLD = 0.75
DEFAULT_TOP_K = 3


def parse_document(doc_path: Path) -> list[str]:
    suffix = doc_path.suffix.lower()
    if suffix == ".pdf":
        pages = extract_pdf_pages_text(doc_path)
        raw_text = clean_text("\n".join(p for p in pages if p.strip()))
    elif suffix in (".txt", ".md"):
        raw_text = clean_text(doc_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: .pdf, .txt, .md")
    return chunk_text(raw_text)


def build_upload_chunks(text_chunks: list[str], doc_path: Path, domain: str) -> list[dict]:
    source_slug = doc_path.stem.lower().replace(" ", "_")[:40]
    year = extract_year(doc_path.name)
    return [
        {
            "chunk_id": f"uploaded_{source_slug}_{i:04d}",
            "chunk_type": "text",
            "domain": domain,
            "source_file": doc_path.name,
            "source_type": "uploaded",
            "doc_type": "uploaded_document",
            "year": year,
            "text": text,
        }
        for i, text in enumerate(text_chunks)
    ]


def check_adversarial(
    chunks: list[dict],
    embedding_model,
    threshold: float = DEFAULT_ADVERSARIAL_THRESHOLD,
) -> tuple[bool, list[str]]:
    adversarial_texts = [
        template["text"]
        for domain in ALLOWED_DOMAINS
        for template in build_adversarial_templates(domain)
    ]

    adversarial_vecs = np.asarray(
        embedding_model.encode(adversarial_texts, normalize_embeddings=True, show_progress_bar=False),
        dtype="float32",
    )

    flagged: list[str] = []
    for chunk in chunks:
        chunk_vec = np.asarray(
            embedding_model.encode([chunk["text"]], normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )[0]
        similarities = adversarial_vecs @ chunk_vec
        max_sim = float(similarities.max())
        if max_sim >= threshold:
            best_match = adversarial_texts[int(similarities.argmax())][:80]
            flagged.append(
                f"Chunk '{chunk['chunk_id']}' scored {max_sim:.2f} similarity to adversarial pattern: "
                f"'{best_match}...'"
            )

    return bool(flagged), flagged


def embed_and_append(chunks: list[dict], embedding_model, store_dir: Path) -> None:
    index_path = store_dir / "rag_index.faiss"
    metadata_path = store_dir / "rag_metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {store_dir}. Run the ingest pipeline first."
        )

    index = faiss.read_index(str(index_path))
    with metadata_path.open("r", encoding="utf-8") as fh:
        existing_metadata: list[dict] = json.load(fh)

    existing_ids = {str(c.get("chunk_id")) for c in existing_metadata}
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    if not new_chunks:
        return

    embeddings = embedding_model.encode(
        [c["text"] for c in new_chunks], normalize_embeddings=True, show_progress_bar=False
    )
    embedding_matrix = np.asarray(embeddings, dtype="float32")

    if index.d != embedding_matrix.shape[1]:
        raise ValueError(
            f"Dimension mismatch: index has {index.d}, embeddings have {embedding_matrix.shape[1]}."
        )

    index.add(embedding_matrix)
    faiss.write_index(index, str(index_path))
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(existing_metadata + new_chunks, fh, ensure_ascii=False, indent=2)


def retrieve_with_priority(
    question: str,
    uploaded_source_file: str,
    top_k: int,
    domain: str | None,
    index_path: Path,
    metadata_path: Path,
    embedding_model,
) -> list[dict]:
    with metadata_path.open("r", encoding="utf-8") as fh:
        all_metadata: list[dict] = json.load(fh)

    query_vec = np.asarray(
        embedding_model.encode([question], normalize_embeddings=True, show_progress_bar=False),
        dtype="float32",
    )
    index = faiss.read_index(str(index_path))
    search_k = min(len(all_metadata), max(top_k * 10, 50))
    _, indices = index.search(query_vec, search_k)

    uploaded: list[dict] = []
    general: list[dict] = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(all_metadata):
            continue
        record = all_metadata[idx]
        if domain and record.get("domain") != domain:
            continue
        if record.get("source_file") == uploaded_source_file:
            uploaded.append(record)
        else:
            general.append(record)

    combined = uploaded[:top_k] + general
    return combined[:top_k]


def run_document_pipeline(
    doc_path: Path,
    question: str,
    domain: str,
    provider: str = "groq",
    model: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    adversarial_threshold: float = DEFAULT_ADVERSARIAL_THRESHOLD,
    store_dir: Path | None = None,
    embedding_model_name: str = DEFAULT_MODEL,
    dry_run: bool = False,
) -> dict:
    store_dir = store_dir or vector_store_dir()

    text_chunks = parse_document(doc_path)
    if not text_chunks:
        return {"status": "error", "reason": "Document produced no text after parsing."}

    chunks = build_upload_chunks(text_chunks, doc_path, domain)

    embedding_model = load_embedding_model(embedding_model_name)
    is_flagged, reasons = check_adversarial(chunks, embedding_model, threshold=adversarial_threshold)
    if is_flagged:
        return {
            "status": "rejected",
            "reason": "Document contains adversarial content and was not embedded.",
            "details": reasons,
        }

    if not dry_run:
        embed_and_append(chunks, embedding_model, store_dir)

    index_path = store_dir / "rag_index.faiss"
    metadata_path = store_dir / "rag_metadata.json"
    retrieved = retrieve_with_priority(
        question=question,
        uploaded_source_file=doc_path.name,
        top_k=top_k,
        domain=domain,
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model,
    )

    context = format_retrieved_context(retrieved)
    prompt = build_survey_response_prompt(question=question, retrieved_context=context)
    response = ""
    if not dry_run:
        response = call_generation_model(prompt=prompt, provider=provider, model=model)

    return {
        "status": "accepted",
        "document": doc_path.name,
        "chunks_embedded": len(chunks),
        "retrieved_sources": [c.get("source_file") for c in retrieved],
        "uploaded_chunks_used": sum(c.get("source_file") == doc_path.name for c in retrieved),
        "response": response,
        "prompt": prompt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a document, run adversarial defense check, embed, and answer a query."
    )
    parser.add_argument("--doc", type=Path, default=None, help="Path to document (.pdf, .txt, .md).")
    parser.add_argument("--question", default=None, help="Query to answer from the document.")
    parser.add_argument("--domain", choices=ALLOWED_DOMAINS, default=None, help="Document domain.")
    parser.add_argument("--provider", default="groq", choices=("groq", "local", "openai"), help="LLM provider.")
    parser.add_argument("--model", default=None, help="Model name for the selected provider.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--adversarial-threshold",
        type=float,
        default=DEFAULT_ADVERSARIAL_THRESHOLD,
        help="Similarity threshold for adversarial detection (0-1). Default: 0.75.",
    )
    parser.add_argument("--store-dir", type=Path, default=None, help="Override vector store directory.")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL, help="Sentence-Transformers model name.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and defense check only — skip embed and LLM.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write result JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.doc is None:
        while True:
            raw = input("Enter path to document (.pdf, .txt, .md): ").strip()
            if not raw:
                print("  Path cannot be empty.")
                continue
            args.doc = Path(raw)
            if not args.doc.exists():
                print(f"  File not found: {args.doc}")
                continue
            break

    if not args.doc.exists():
        print(f"ERROR: Document not found: {args.doc}")
        raise SystemExit(1)

    if args.question is None:
        while True:
            args.question = input("Enter your question: ").strip()
            if args.question:
                break
            print("  Question cannot be empty.")

    if args.domain is None:
        while True:
            args.domain = input(f"Enter domain ({'/'.join(ALLOWED_DOMAINS)}): ").strip()
            if args.domain in ALLOWED_DOMAINS:
                break
            print(f"  Invalid domain. Choose from: {', '.join(ALLOWED_DOMAINS)}")

    result = run_document_pipeline(
        doc_path=args.doc,
        question=args.question,
        domain=args.domain,
        provider=args.provider,
        model=args.model,
        top_k=args.top_k,
        adversarial_threshold=args.adversarial_threshold,
        store_dir=args.store_dir,
        embedding_model_name=args.embedding_model,
        dry_run=args.dry_run,
    )

    if args.output_json:
        write_json(args.output_json, result)
        print(f"Result written to {args.output_json}")

    status = result["status"]
    if status == "error":
        print(f"\nERROR: {result['reason']}")
    elif status == "rejected":
        print(f"\nDOCUMENT REJECTED — adversarial content detected:")
        for detail in result.get("details", []):
            print(f"  • {detail}")
    else:
        if args.dry_run:
            print(f"\nDocument accepted (dry run — not embedded) ({result['chunks_embedded']} chunks would be added)")
        else:
            print(f"\nDocument accepted and embedded ({result['chunks_embedded']} chunks)")
        print(f"Retrieved sources: {result['retrieved_sources']}")
        print(f"  (uploaded doc chunks used: {result['uploaded_chunks_used']} of {args.top_k})")
        if not args.dry_run:
            print(f"\nAnswer:\n{result['response']}")


if __name__ == "__main__":
    main()
