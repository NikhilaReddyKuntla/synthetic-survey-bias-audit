from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from src.rag.embed import DEFAULT_MODEL, load_embedding_model
from src.utils.helpers import rag_validation_outputs_dir, write_json, vector_store_dir

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "about",
    "affect",
    "after",
    "also",
    "answer",
    "because",
    "before",
    "being",
    "between",
    "customer",
    "customers",
    "does",
    "from",
    "have",
    "this",
    "that",
    "their",
    "there",
    "these",
    "they",
    "those",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "survey",
    "question",
}
BOILERPLATE_PATTERNS = (
    "survey design and methods",
    "survey methodology",
    "response rate",
    "sampling",
    "sample design",
    "post-stratification",
    "suggested citation",
    "notes:",
)


def load_vector_store(
    index_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[faiss.Index, list[dict]]:
    import faiss

    store_dir = vector_store_dir()
    index_path = index_path or (store_dir / "rag_index.faiss")
    metadata_path = metadata_path or (store_dir / "rag_metadata.json")

    index = faiss.read_index(str(index_path))
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    return index, metadata


def tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text or "")]
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def _record_search_text(record: dict) -> str:
    fields = (
        record.get("domain"),
        record.get("doc_type"),
        record.get("source_file"),
        record.get("year"),
        record.get("chunk_type"),
        record.get("text"),
    )
    return " ".join(str(field) for field in fields if field)


def lexical_relevance_score(query: str, record: dict) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    record_tokens = tokenize(_record_search_text(record))
    if not record_tokens:
        return 0.0

    record_counts: dict[str, int] = {}
    for token in record_tokens:
        record_counts[token] = record_counts.get(token, 0) + 1

    unique_query_tokens = set(query_tokens)
    matched = sum(1 for token in unique_query_tokens if token in record_counts)
    frequency_bonus = sum(math.log1p(record_counts.get(token, 0)) for token in unique_query_tokens)
    coverage = matched / len(unique_query_tokens)
    return float(coverage + (0.08 * frequency_bonus))


def _semantic_search_candidates(
    scores: np.ndarray,
    indices: np.ndarray,
    metadata: list[dict],
    domain: str | None,
    min_score: float | None,
) -> dict[int, float]:
    candidates: dict[int, float] = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        record = metadata[idx]
        if domain and record.get("domain") != domain:
            continue
        if min_score is not None and float(score) < min_score:
            continue
        candidates[int(idx)] = float(score)
    return candidates


def boilerplate_penalty(record: dict) -> float:
    text = str(record.get("text") or "").lower()
    if not text:
        return 0.0
    matches = sum(1 for pattern in BOILERPLATE_PATTERNS if pattern in text)
    return float(min(0.18, matches * 0.06))


def _lexical_search_candidates(
    query: str,
    metadata: list[dict],
    domain: str | None,
    limit: int,
) -> list[tuple[int, float]]:
    scored: list[tuple[float, int, int]] = []
    for position, record in enumerate(metadata):
        if domain and record.get("domain") != domain:
            continue
        lexical_score = lexical_relevance_score(query, record)
        if lexical_score <= 0:
            continue
        scored.append((lexical_score, -position, position))

    scored.sort(reverse=True)
    return [(position, score) for score, _, position in scored[:limit]]


def _is_near_duplicate(candidate: dict, selected: list[dict], threshold: float = 0.92) -> bool:
    candidate_tokens = set(tokenize(str(candidate.get("text") or "")))
    if not candidate_tokens:
        return False

    for existing in selected:
        existing_tokens = set(tokenize(str(existing.get("text") or "")))
        if not existing_tokens:
            continue
        overlap = len(candidate_tokens & existing_tokens) / len(candidate_tokens | existing_tokens)
        if overlap >= threshold:
            return True
    return False


def retrieve_chunks(
    query: str,
    top_k: int = 8,
    domain: str | None = None,
    model_name: str = DEFAULT_MODEL,
    embedding_model=None,
    index_path: Path | None = None,
    metadata_path: Path | None = None,
    min_score: float | None = 0.18,
    candidate_multiplier: int = 12,
    semantic_weight: float = 0.72,
    lexical_weight: float = 0.28,
    dedupe_near_duplicates: bool = True,
    max_chunks_per_source: int | None = 3,
) -> list[dict]:
    model = embedding_model or load_embedding_model(model_name)
    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    index, metadata = load_vector_store(index_path=index_path, metadata_path=metadata_path)

    search_k = len(metadata) if domain else min(len(metadata), max(top_k * candidate_multiplier, top_k))
    scores, indices = index.search(query_vector, search_k)

    candidate_scores = _semantic_search_candidates(
        scores=scores,
        indices=indices,
        metadata=metadata,
        domain=domain,
        min_score=min_score,
    )
    lexical_limit = min(len(metadata), max(top_k * candidate_multiplier, top_k))
    for idx, _ in _lexical_search_candidates(query=query, metadata=metadata, domain=domain, limit=lexical_limit):
        candidate_scores.setdefault(idx, 0.0)

    lexical_scores = {
        idx: lexical_relevance_score(query, metadata[idx])
        for idx in candidate_scores
    }
    max_lexical_score = max(lexical_scores.values(), default=0.0) or 1.0

    reranked: list[tuple[float, float, float, int, dict]] = []
    for idx, semantic_score in candidate_scores.items():
        lexical_score = lexical_scores[idx]
        if min_score is not None and semantic_score < min_score and lexical_score <= 0:
            continue
        normalized_lexical = lexical_score / max_lexical_score
        quality_penalty = boilerplate_penalty(metadata[idx])
        hybrid_score = (semantic_weight * semantic_score) + (lexical_weight * normalized_lexical) - quality_penalty
        record = dict(metadata[idx])
        record["similarity_score"] = float(semantic_score)
        record["lexical_score"] = float(lexical_score)
        record["quality_penalty"] = float(quality_penalty)
        record["hybrid_score"] = float(hybrid_score)
        reranked.append((hybrid_score, semantic_score, lexical_score, -idx, record))

    reranked.sort(reverse=True)

    results: list[dict] = []
    source_counts: dict[str, int] = {}
    for _, _, _, _, record in reranked:
        if dedupe_near_duplicates and _is_near_duplicate(record, results):
            continue
        source_file = str(record.get("source_file") or "")
        if max_chunks_per_source is not None and source_file:
            if source_counts.get(source_file, 0) >= max_chunks_per_source:
                continue
            source_counts[source_file] = source_counts.get(source_file, 0) + 1
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
    parser.add_argument("--top-k", type=int, default=8, help="Number of chunks to return.")
    parser.add_argument("--domain", default=None, help="Optional domain filter.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformers model name.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.18,
        help="Minimum cosine/inner-product similarity score. Use 0 to disable filtering.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=12,
        help="Retrieve and rerank this many candidates per requested top-k result.",
    )
    parser.add_argument("--semantic-weight", type=float, default=0.72, help="Hybrid reranking weight for vector similarity.")
    parser.add_argument("--lexical-weight", type=float, default=0.28, help="Hybrid reranking weight for keyword coverage.")
    parser.add_argument(
        "--allow-near-duplicates",
        action="store_true",
        help="Keep near-duplicate chunks instead of suppressing them during reranking.",
    )
    parser.add_argument(
        "--max-chunks-per-source",
        type=int,
        default=3,
        help="Maximum chunks to return from one source file. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=rag_validation_outputs_dir() / "retrieval_results.json",
        help="Optional JSON output path for saving retrieval validation results.",
    )
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
        min_score=args.min_score if args.min_score > 0 else None,
        candidate_multiplier=args.candidate_multiplier,
        semantic_weight=args.semantic_weight,
        lexical_weight=args.lexical_weight,
        dedupe_near_duplicates=not args.allow_near_duplicates,
        max_chunks_per_source=args.max_chunks_per_source if args.max_chunks_per_source > 0 else None,
    )
    if args.output:
        write_json(args.output, {"query": args.query, "domain": args.domain, "top_k": args.top_k, "results": results})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
