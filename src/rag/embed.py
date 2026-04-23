from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.helpers import ensure_dir, rag_docs_path, read_jsonl, vector_store_dir

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        try:
            return SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            raise RuntimeError(
                f"Unable to load embedding model '{model_name}'. "
                "If this is the first run, make sure the environment can download the model once."
            ) from exc


def embed_chunks(
    chunks_path: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    output_dir: Path | None = None,
    append: bool = False,
) -> tuple[Path, Path]:
    chunks_path = chunks_path or rag_docs_path()
    output_dir = output_dir or vector_store_dir()
    ensure_dir(output_dir)

    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    index_path = output_dir / "rag_index.faiss"
    metadata_path = output_dir / "rag_metadata.json"

    existing_metadata: list[dict] = []
    index: faiss.Index | None = None
    if append and index_path.exists() and metadata_path.exists():
        index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing_metadata = json.load(handle)

        existing_ids = {chunk["chunk_id"] for chunk in existing_metadata}
        chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in existing_ids]
        if not chunks:
            return index_path, metadata_path

    model = load_embedding_model(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embedding_matrix = np.asarray(embeddings, dtype="float32")

    if index is None:
        index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    elif index.d != embedding_matrix.shape[1]:
        raise ValueError(
            f"Existing FAISS index dimension {index.d} does not match "
            f"new embedding dimension {embedding_matrix.shape[1]}."
        )

    index.add(embedding_matrix)
    faiss.write_index(index, str(index_path))
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(existing_metadata + chunks, handle, ensure_ascii=False, indent=2)

    return index_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed RAG chunks into a FAISS index.")
    parser.add_argument("--chunks", type=Path, default=rag_docs_path(), help="Path to chunk JSONL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformers model name.")
    parser.add_argument("--output-dir", type=Path, default=vector_store_dir(), help="Vector store output directory.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add only new chunk IDs to an existing FAISS index and metadata file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path, metadata_path = embed_chunks(
        chunks_path=args.chunks,
        model_name=args.model,
        output_dir=args.output_dir,
        append=args.append,
    )
    print(f"Wrote FAISS index to {index_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
