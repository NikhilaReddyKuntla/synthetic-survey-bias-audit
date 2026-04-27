from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from src.adversarial.defense_decision import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MIN_CONFIDENCE,
    DEFAULT_JUDGE_TIMEOUT,
    DEFAULT_LOCAL_ENDPOINT,
    GENERATION_PROVIDERS,
    evaluate_defense_candidate,
    group_trusted_chunks_by_domain,
)
from src.rag.embed import DEFAULT_MODEL, load_embedding_model
from src.rag.ingest import ALLOWED_DOMAINS, infer_doc_type, slugify
from src.utils.doc_utils import extract_clean_document_text, supported_document
from src.utils.helpers import (
    append_jsonl,
    ensure_dir,
    read_json,
    user_uploads_dir,
    user_upload_outputs_dir,
    user_validated_chunks_path,
    user_vector_store_dir,
    vector_store_dir,
    write_json,
)
from src.utils.text_utils import chunk_text, extract_year


def _normalize_input_path(path: Path) -> Path:
    if path.exists():
        return path
    normalized = Path(str(path).replace("\\", "/"))
    return normalized if normalized.exists() else path


def _upload_id(path: Path, purpose: str | None) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    digest.update(str(purpose or "").encode("utf-8"))
    return digest.hexdigest()[:16]


def _copy_upload(path: Path, upload_id: str) -> Path:
    ensure_dir(user_uploads_dir())
    destination = user_uploads_dir() / f"{upload_id}_{path.name}"
    if path.resolve() != destination.resolve():
        shutil.copy2(path, destination)
    return destination


def validate_uploaded_text(
    text: str,
    trusted_chunks: list[dict],
    domain: str | None = None,
    provider: str = "groq",
    model: str | None = None,
    judge_model: str | None = None,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    judge_timeout: int = DEFAULT_JUDGE_TIMEOUT,
    judge_min_confidence: float = DEFAULT_JUDGE_MIN_CONFIDENCE,
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> dict:
    decision = evaluate_defense_candidate(
        text=text,
        trusted_chunks_by_domain=group_trusted_chunks_by_domain(trusted_chunks),
        domain=domain,
        target_claim=None,
        provider=provider,
        model=model,
        judge_model=judge_model,
        local_endpoint=local_endpoint,
        judge_timeout=judge_timeout,
        judge_min_confidence=judge_min_confidence,
        judge_max_tokens=judge_max_tokens,
        run_judge=True,
        fail_closed=True,
        check_prompt_injection=True,
    )
    trust_score = decision["final_trust_score"]
    return {
        "trust_score": trust_score,
        "final_trust_score": trust_score,
        "defense_passed": decision["defense_passed"],
        "accepted": decision["defense_passed"],
        "static_score": decision["static_score"],
        "support_score": decision["support_score"],
        "judge_verdict": decision["judge_verdict"],
        "judge_confidence": decision["judge_confidence"],
        "judge_reason": decision["judge_reason"],
        "judge_failed": decision["judge_failed"],
        "judge_error": decision["judge_error"],
        "has_prompt_injection": decision["has_prompt_injection"],
        "has_unverified_stats": decision["has_unverified_stats"],
        "has_absolute_language": decision["has_absolute_language"],
        "low_alignment": decision["low_alignment"],
        "reasons": decision["reasons"],
        "recommended_action": "index_for_retrieval" if decision["defense_passed"] else "reject_upload",
        "defense_version": decision["defense_version"],
    }


def build_user_upload_chunks(
    text: str,
    source_path: Path,
    upload_id: str,
    validation: dict,
    domain: str | None = None,
    purpose: str | None = None,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[dict]:
    domain_label = domain or "general"
    source_slug = slugify(source_path.stem)
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    year = extract_year(source_path.name)
    doc_type = infer_doc_type(source_path.name)

    records: list[dict] = []
    for index, chunk in enumerate(chunks, start=1):
        records.append(
            {
                "chunk_id": f"user_{upload_id}_{source_slug}_{index:03d}",
                "chunk_type": "text",
                "domain": domain_label,
                "source_file": source_path.name,
                "source_kind": "user_upload",
                "doc_type": doc_type,
                "year": year,
                "upload_id": upload_id,
                "upload_purpose": purpose,
                "trust_score": validation["trust_score"],
                "final_trust_score": validation["final_trust_score"],
                "defense_passed": validation["defense_passed"],
                "defense_version": validation["defense_version"],
                "validation_reasons": validation["reasons"],
                "text": chunk,
            }
        )
    return records


def _append_chunks_to_user_vector_store(chunks: list[dict], model_name: str = DEFAULT_MODEL) -> dict:
    if not chunks:
        raise ValueError("Cannot index an empty chunk list.")

    store_dir = user_vector_store_dir()
    ensure_dir(store_dir)
    index_path = store_dir / "rag_index.faiss"
    metadata_path = store_dir / "rag_metadata.json"

    existing_metadata: list[dict] = []
    index: faiss.Index | None = None
    existing_ids: set[str] = set()
    if index_path.exists() and metadata_path.exists():
        index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing_metadata = json.load(handle)
        existing_ids = {str(record.get("chunk_id")) for record in existing_metadata if record.get("chunk_id")}

    new_chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in existing_ids]
    if not new_chunks:
        return {"index_path": index_path, "metadata_path": metadata_path, "indexed_chunks": 0}

    model = load_embedding_model(model_name)
    embeddings = model.encode([chunk["text"] for chunk in new_chunks], normalize_embeddings=True, show_progress_bar=True)
    embedding_matrix = np.asarray(embeddings, dtype="float32")

    if index is None:
        index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    elif index.d != embedding_matrix.shape[1]:
        raise ValueError(
            f"Existing user FAISS index dimension {index.d} does not match "
            f"new embedding dimension {embedding_matrix.shape[1]}."
        )

    index.add(embedding_matrix)
    faiss.write_index(index, str(index_path))
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(existing_metadata + new_chunks, handle, ensure_ascii=False, indent=2)

    append_jsonl(user_validated_chunks_path(), new_chunks)
    return {"index_path": index_path, "metadata_path": metadata_path, "indexed_chunks": len(new_chunks)}


def validate_and_index_documents(
    input_paths: list[Path],
    domain: str | None = None,
    purpose: str | None = None,
    trusted_metadata_path: Path | None = None,
    provider: str = "groq",
    model: str | None = None,
    judge_model: str | None = None,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    judge_timeout: int = DEFAULT_JUDGE_TIMEOUT,
    judge_min_confidence: float = DEFAULT_JUDGE_MIN_CONFIDENCE,
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    model_name: str = DEFAULT_MODEL,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    report_output: Path | None = None,
) -> dict:
    trusted_metadata_path = trusted_metadata_path or (vector_store_dir() / "rag_metadata.json")
    trusted_chunks = read_json(trusted_metadata_path) if trusted_metadata_path.exists() else []
    if not isinstance(trusted_chunks, list):
        raise ValueError(f"Expected trusted metadata JSON list in {trusted_metadata_path}")

    accepted_documents: list[dict] = []
    rejected_documents: list[dict] = []
    accepted_chunks: list[dict] = []

    for input_path in input_paths:
        path = _normalize_input_path(Path(input_path))
        if not path.exists():
            raise FileNotFoundError(f"Uploaded document not found: {path}")
        if not supported_document(path):
            raise ValueError(f"Unsupported uploaded document type: {path}")

        text = extract_clean_document_text(path)
        if not text:
            raise ValueError(f"No extractable text found in uploaded document: {path}")

        upload_id = _upload_id(path, purpose=purpose)
        stored_path = _copy_upload(path, upload_id=upload_id)
        validation = validate_uploaded_text(
            text=text,
            trusted_chunks=trusted_chunks,
            domain=domain,
            provider=provider,
            model=model,
            judge_model=judge_model,
            local_endpoint=local_endpoint,
            judge_timeout=judge_timeout,
            judge_min_confidence=judge_min_confidence,
            judge_max_tokens=judge_max_tokens,
        )

        document_record = {
            "upload_id": upload_id,
            "source_file": path.name,
            "stored_path": str(stored_path),
            "domain": domain or "general",
            "upload_purpose": purpose,
            "validation": validation,
        }
        if validation["accepted"]:
            chunks = build_user_upload_chunks(
                text=text,
                source_path=path,
                upload_id=upload_id,
                validation=validation,
                domain=domain,
                purpose=purpose,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            accepted_chunks.extend(chunks)
            document_record["chunk_count"] = len(chunks)
            accepted_documents.append(document_record)
        else:
            rejected_documents.append(document_record)

    index_result = None
    if accepted_chunks:
        index_result = _append_chunks_to_user_vector_store(chunks=accepted_chunks, model_name=model_name)

    trust_counter = Counter(record["validation"]["trust_score"] for record in accepted_documents + rejected_documents)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_documents": len(accepted_documents) + len(rejected_documents),
            "accepted_documents": len(accepted_documents),
            "rejected_documents": len(rejected_documents),
            "indexed_chunks": len(accepted_chunks),
            "trust_distribution": {
                "high": trust_counter.get("high", 0),
                "medium": trust_counter.get("medium", 0),
                "low": trust_counter.get("low", 0),
            },
        },
        "accepted_documents": accepted_documents,
        "rejected_documents": rejected_documents,
        "index_result": {
            "index_path": str(index_result["index_path"]),
            "metadata_path": str(index_result["metadata_path"]),
            "indexed_chunks": index_result["indexed_chunks"],
        }
        if index_result
        else None,
    }

    report_output = report_output or (user_upload_outputs_dir() / "user_upload_validation_report.json")
    write_json(report_output, report)
    report["report_path"] = str(report_output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate user-uploaded documents and index accepted chunks.")
    parser.add_argument("--input", type=Path, action="append", required=True, dest="input_paths")
    parser.add_argument("--domain", choices=ALLOWED_DOMAINS, help="Optional domain label for uploaded documents.")
    parser.add_argument("--purpose", help="Purpose label used to prioritize this upload during retrieval.")
    parser.add_argument(
        "--trusted-metadata-path",
        type=Path,
        default=vector_store_dir() / "rag_metadata.json",
        help="Trusted clean metadata used by the defense checks.",
    )
    parser.add_argument("--provider", default="groq", choices=GENERATION_PROVIDERS, help="Judge provider.")
    parser.add_argument("--model", default=None, help="Judge provider model unless --judge-model is set.")
    parser.add_argument("--judge-model", default=None, help="Optional dedicated judge model.")
    parser.add_argument("--local-endpoint", default=DEFAULT_LOCAL_ENDPOINT, help="Local provider endpoint.")
    parser.add_argument("--judge-timeout", type=int, default=DEFAULT_JUDGE_TIMEOUT, help="Judge timeout in seconds.")
    parser.add_argument(
        "--judge-min-confidence",
        type=float,
        default=DEFAULT_JUDGE_MIN_CONFIDENCE,
        help="Minimum judge confidence required for indexing.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_TOKENS,
        help="Maximum completion tokens for judge response.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL, help="Embedding model for accepted upload chunks.")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=user_upload_outputs_dir() / "user_upload_validation_report.json",
        help="Validation report output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_and_index_documents(
        input_paths=args.input_paths,
        domain=args.domain,
        purpose=args.purpose,
        trusted_metadata_path=args.trusted_metadata_path,
        provider=args.provider,
        model=args.model,
        judge_model=args.judge_model,
        local_endpoint=args.local_endpoint,
        judge_timeout=args.judge_timeout,
        judge_min_confidence=args.judge_min_confidence,
        judge_max_tokens=args.judge_max_tokens,
        model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        report_output=args.output,
    )
    summary = report["summary"]
    print(
        "Upload validation complete:"
        f" accepted={summary['accepted_documents']}, rejected={summary['rejected_documents']}, "
        f"indexed_chunks={summary['indexed_chunks']}."
    )
    print(f"Wrote upload validation report to {report['report_path']}")


if __name__ == "__main__":
    main()
