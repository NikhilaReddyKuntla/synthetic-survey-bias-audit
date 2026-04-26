from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from src.rag.embed import DEFAULT_MODEL, load_embedding_model
from src.utils.helpers import vector_store_dir


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


def retrieve_chunks(
    query: str,
    top_k: int = 3,
    domain: str | None = None,
    model_name: str = DEFAULT_MODEL,
    index_path: Path | None = None,
    metadata_path: Path | None = None,
) -> list[dict]:
    index, metadata = load_vector_store(index_path=index_path, metadata_path=metadata_path)
    model = load_embedding_model(model_name)
    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")

    search_k = len(metadata) if domain else min(len(metadata), max(top_k * 5, top_k))
    _, indices = index.search(query_vector, search_k)

    results: list[dict] = []
    for idx in indices[0]:
        if idx < 0:
            continue
        record = metadata[idx]
        if domain and record["domain"] != domain:
            continue
        results.append(record)
        if len(results) >= top_k:
            break

    return results


def format_retrieved_context(results: list[dict]) -> str:
    blocks: list[str] = []
    for result in results:
        page = f" | Page: {result['page_number']}" if result.get("page_number") else ""
        header = (
            f"[Source: {result['source_file']} | Domain: {result['domain']} | "
            f"Year: {result.get('year')} | Chunk Type: {result.get('chunk_type', 'text')}{page}]"
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
