from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from src.adversarial.validate_docs import _jaccard_similarity, _group_trusted_chunks_by_domain
from src.attacks.poison_utils import tokenize
from src.rag.embed import DEFAULT_MODEL, load_embedding_model
from src.rag.ingest import ALLOWED_DOMAINS, infer_doc_type, slugify
from src.utils.doc_utils import extract_clean_document_text, supported_document
from src.utils.helpers import (
    append_jsonl,
    ensure_dir,
    outputs_dir,
    read_json,
    user_uploads_dir,
    user_validated_chunks_path,
    user_vector_store_dir,
    vector_store_dir,
    write_json,
)
from src.utils.text_utils import chunk_text, extract_year

PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore (all )?(previous|prior|above) instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard (all )?(previous|prior|above) instructions\b", re.IGNORECASE),
    re.compile(r"\breveal (the )?(system|developer) (prompt|message|instructions)\b", re.IGNORECASE),
    re.compile(r"\b(system|developer) prompt\b", re.IGNORECASE),
    re.compile(r"\bapi[_ -]?key\b", re.IGNORECASE),
    re.compile(r"\bdo not follow\b.*\buser\b", re.IGNORECASE),
    re.compile(r"\boverride\b.*\b(instructions|policy|rules)\b", re.IGNORECASE),
)
STAT_PATTERN = re.compile(r"\b\d{2,}%\b|\b\d{3,}\s+respondents?\b", re.IGNORECASE)
ABSOLUTE_PATTERN = re.compile(r"\b(all|always|never|every|guaranteed|undeniable)\b", re.IGNORECASE)
LOW_ALIGNMENT_THRESHOLD = 0.045
LOW_SUPPORT_THRESHOLD = 0.08


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


def _trusted_token_sets(trusted_chunks: list[dict], domain: str | None = None) -> list[set[str]]:
    grouped = _group_trusted_chunks_by_domain(trusted_chunks)
    if domain:
        records = grouped.get(domain, [])
    else:
        records = [chunk for chunks in grouped.values() for chunk in chunks]
    if not records:
        records = [chunk for chunks in grouped.values() for chunk in chunks]
    return [tokenize(str(chunk.get("text") or "")) for chunk in records if isinstance(chunk, dict)]


def _max_support(text: str, trusted_token_sets: list[set[str]]) -> float:
    tokens = tokenize(text)
    return max((_jaccard_similarity(tokens, trusted_tokens) for trusted_tokens in trusted_token_sets), default=0.0)


def validate_uploaded_text(text: str, trusted_chunks: list[dict], domain: str | None = None) -> dict:
    trusted_token_sets = _trusted_token_sets(trusted_chunks=trusted_chunks, domain=domain)
    support_score = _max_support(text, trusted_token_sets)
    has_prompt_injection = any(pattern.search(text) for pattern in PROMPT_INJECTION_PATTERNS)
    has_unverified_stats = bool(STAT_PATTERN.search(text)) and support_score < LOW_SUPPORT_THRESHOLD
    has_absolute_language = bool(ABSOLUTE_PATTERN.search(text)) and support_score < LOW_SUPPORT_THRESHOLD
    low_alignment = support_score < LOW_ALIGNMENT_THRESHOLD

    reasons: list[str] = []
    if has_prompt_injection:
        reasons.append("prompt_injection_or_instruction_override")
    if has_unverified_stats:
        reasons.append("unsupported_statistical_claim")
    if has_absolute_language:
        reasons.append("unsupported_absolute_language")
    if low_alignment:
        reasons.append("low_alignment_with_trusted_chunks")

    if has_prompt_injection or has_unverified_stats or has_absolute_language:
        trust_score = "low"
    elif low_alignment:
        trust_score = "medium"
    else:
        trust_score = "high"

    return {
        "trust_score": trust_score,
        "accepted": trust_score != "low",
        "support_score": support_score,
        "has_prompt_injection": has_prompt_injection,
        "has_unverified_stats": has_unverified_stats,
        "has_absolute_language": has_absolute_language,
        "low_alignment": low_alignment,
        "reasons": reasons,
        "recommended_action": "index_for_retrieval" if trust_score != "low" else "reject_upload",
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
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Uploaded document not found: {path}")
        if not supported_document(path):
            raise ValueError(f"Unsupported uploaded document type: {path}")

        text = extract_clean_document_text(path)
        if not text:
            raise ValueError(f"No extractable text found in uploaded document: {path}")

        upload_id = _upload_id(path, purpose=purpose)
        stored_path = _copy_upload(path, upload_id=upload_id)
        validation = validate_uploaded_text(text=text, trusted_chunks=trusted_chunks, domain=domain)

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

    report_output = report_output or (outputs_dir() / "user_upload_validation_report.json")
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
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL, help="Embedding model for accepted upload chunks.")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=outputs_dir() / "user_upload_validation_report.json",
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
