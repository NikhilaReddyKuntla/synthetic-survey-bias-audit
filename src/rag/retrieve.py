from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.embed import DEFAULT_MODEL, load_embedding_model
from src.utils.helpers import user_vector_store_dir, vector_store_dir


def load_vector_store(
    index_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[faiss.Index, list[dict]]:
    store_dir = vector_store_dir()
    index_path = index_path or (store_dir / "rag_index.faiss")
    metadata_path = metadata_path or (store_dir / "rag_metadata.json")

    index = faiss.read_index(str(index_path))
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    return index, metadata


def _matches_metadata_filters(record: dict[str, Any], filters: dict[str, Any] | None = None) -> bool:
    if not filters:
        return True

    for key, expected in filters.items():
        value = record.get(key)
        if expected is None:
            if value is not None:
                return False
            continue
        if isinstance(expected, (set, list, tuple)):
            if value not in expected:
                return False
            continue
        if value != expected:
            return False

    return True


def search_vector_store(
    query: str,
    top_k: int = 3,
    domain: str | None = None,
    model_name: str = DEFAULT_MODEL,
    embedding_model: SentenceTransformer | None = None,
    index_path: Path | None = None,
    metadata_path: Path | None = None,
    metadata_filters: dict[str, Any] | None = None,
    min_score: float | None = None,
) -> list[dict]:
    if top_k <= 0:
        return []

    index, metadata = load_vector_store(index_path=index_path, metadata_path=metadata_path)
    model = embedding_model or load_embedding_model(model_name)
    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")

    scores, indices = index.search(query_vector, len(metadata))

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0], strict=False):
        if idx < 0:
            continue
        record = metadata[idx]
        if domain and record["domain"] != domain:
            continue
        if not _matches_metadata_filters(record, metadata_filters):
            continue
        if min_score is not None and float(score) < min_score:
            continue

        enriched = dict(record)
        enriched["score"] = float(score)
        results.append(enriched)
        if len(results) >= top_k:
            break

    return results


def retrieve_chunks(
    query: str,
    top_k: int = 3,
    domain: str | None = None,
    model_name: str = DEFAULT_MODEL,
    embedding_model: SentenceTransformer | None = None,
    index_path: Path | None = None,
    metadata_path: Path | None = None,
) -> list[dict]:
    results = search_vector_store(
        query=query,
        top_k=top_k,
        domain=domain,
        model_name=model_name,
        embedding_model=embedding_model,
        index_path=index_path,
        metadata_path=metadata_path,
    )
    return [{key: value for key, value in result.items() if key != "score"} for result in results]


def retrieve_chunks_with_priority(
    query: str,
    top_k: int = 3,
    domain: str | None = None,
    model_name: str = DEFAULT_MODEL,
    embedding_model: SentenceTransformer | None = None,
    user_index_path: Path | None = None,
    user_metadata_path: Path | None = None,
    upload_purpose: str | None = None,
    upload_ids: list[str] | None = None,
    user_min_score: float = 0.35,
) -> list[dict]:
    combined_results: list[dict] = []
    seen_chunk_ids: set[str] = set()

    default_user_index = user_index_path or (user_vector_store_dir() / "rag_index.faiss")
    default_user_metadata = user_metadata_path or (user_vector_store_dir() / "rag_metadata.json")
    user_store_exists = default_user_index.exists() and default_user_metadata.exists()

    if user_store_exists:
        user_filters: dict[str, Any] = {"source_kind": "user_upload"}
        if upload_purpose:
            user_filters["upload_purpose"] = upload_purpose
        if upload_ids:
            user_filters["upload_id"] = set(upload_ids)

        user_results = search_vector_store(
            query=query,
            top_k=top_k,
            domain=domain,
            model_name=model_name,
            embedding_model=embedding_model,
            index_path=default_user_index,
            metadata_path=default_user_metadata,
            metadata_filters=user_filters,
            min_score=user_min_score,
        )
        for result in user_results:
            if result["chunk_id"] in seen_chunk_ids:
                continue
            seen_chunk_ids.add(result["chunk_id"])
            combined_results.append(result)
            if len(combined_results) >= top_k:
                break

    if len(combined_results) < top_k:
        trusted_results = search_vector_store(
            query=query,
            top_k=max(top_k * 3, top_k),
            domain=domain,
            model_name=model_name,
            embedding_model=embedding_model,
        )
        for result in trusted_results:
            if result["chunk_id"] in seen_chunk_ids:
                continue
            seen_chunk_ids.add(result["chunk_id"])
            combined_results.append(result)
            if len(combined_results) >= top_k:
                break

    return [{key: value for key, value in result.items() if key != "score"} for result in combined_results]


def format_retrieved_context(results: list[dict]) -> str:
    blocks: list[str] = []
    for result in results:
        page = f" | Page: {result['page_number']}" if result.get("page_number") else ""
        source_kind = result.get("source_kind")
        trust = result.get("trust_score")
        source_kind_block = f" | Source Kind: {source_kind}" if source_kind else ""
        trust_block = f" | Trust: {trust}" if trust else ""
        header = (
            f"[Source: {result['source_file']} | Domain: {result['domain']} | "
            f"Year: {result.get('year')} | Chunk Type: {result.get('chunk_type', 'text')}"
            f"{source_kind_block}{trust_block}{page}]"
        )
        blocks.append(f"{header}\n{result['text']}")
    return "\n\n".join(blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve the most relevant RAG chunks for a query.")
    parser.add_argument("--query", required=True, help="Query text to retrieve against.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to return.")
    parser.add_argument("--domain", default=None, help="Optional domain filter.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformers model name.")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=vector_store_dir() / "rag_index.faiss",
        help="Path to the FAISS index.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=vector_store_dir() / "rag_metadata.json",
        help="Path to the metadata JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = retrieve_chunks(
        query=args.query,
        top_k=args.top_k,
        domain=args.domain,
        model_name=args.model,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
