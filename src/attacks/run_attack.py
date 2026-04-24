from __future__ import annotations

import argparse
from pathlib import Path

from src.attacks.poison_utils import (
    create_poisoned_vector_store,
    evaluate_response_shift,
    retrieve_chunks_lexical,
    write_rows_to_csv,
)
from src.generation.generate_responses import (
    DEFAULT_LOCAL_ENDPOINT,
    call_generation_model,
    load_persona,
    load_personas,
    load_questions,
    source_refs,
)
from src.rag.embed import DEFAULT_MODEL, load_embedding_model
from src.rag.retrieve import format_retrieved_context, retrieve_chunks
from src.utils.helpers import personas_path, project_root, read_json, vector_store_dir, write_json
from src.utils.prompt_templates import build_survey_response_prompt

ALLOWED_DOMAINS = ("ecommerce", "finance", "healthcare")


def default_results_dir() -> Path:
    return project_root() / "results"


def _model_label(provider: str, model: str | None) -> str | None:
    return model if model else f"default:{provider}"


def generate_response_with_store(
    question: str,
    persona: dict,
    index_path: Path,
    metadata_path: Path,
    metadata_records: list[dict] | None = None,
    top_k: int = 3,
    domain: str | None = None,
    provider: str = "groq",
    model: str | None = None,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    dry_run: bool = False,
    fast_retrieval: bool = False,
) -> dict:
    if fast_retrieval:
        metadata_payload = metadata_records if metadata_records is not None else read_json(metadata_path)
        if not isinstance(metadata_payload, list):
            raise ValueError(f"Expected metadata JSON array at {metadata_path}")
        chunks = retrieve_chunks_lexical(
            query=question,
            metadata=metadata_payload,
            top_k=top_k,
            domain=domain,
        )
    else:
        chunks = retrieve_chunks(
            query=question,
            top_k=top_k,
            domain=domain,
            index_path=index_path,
            metadata_path=metadata_path,
        )
    retrieved_context = format_retrieved_context(chunks)
    prompt = build_survey_response_prompt(
        question=question,
        retrieved_context=retrieved_context,
        persona=persona,
    )
    response_text = ""
    if not dry_run:
        response_text = call_generation_model(
            prompt=prompt,
            provider=provider,
            model=model,
            local_endpoint=local_endpoint,
        )

    return {
        "prompt": prompt,
        "response": response_text,
        "retrieved_sources": source_refs(chunks),
    }


def resolve_personas(args: argparse.Namespace) -> list[dict]:
    if args.persona_json or args.persona_file:
        persona = load_persona(persona_json=args.persona_json, persona_file=args.persona_file)
        personas = [persona] if persona else []
    else:
        personas = load_personas(args.personas)

    if args.limit_personas is not None:
        personas = personas[: args.limit_personas]
    if not personas:
        raise ValueError("No personas available for attack run.")
    return personas


def parse_args() -> argparse.Namespace:
    results_dir = default_results_dir()
    parser = argparse.ArgumentParser(description="Run baseline vs poisoned RAG responses and quantify attack shift.")
    parser.add_argument("--question", action="append", help="Survey question. Can be provided multiple times.")
    parser.add_argument("--questions-file", type=Path, help="Text file (one per line) or JSON list of questions.")
    parser.add_argument("--domain", choices=ALLOWED_DOMAINS, help="Optional retrieval domain filter.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve.")
    parser.add_argument("--persona-json", help="Optional persona as a JSON object string.")
    parser.add_argument("--persona-file", type=Path, help="Optional JSONL persona file. Uses first record.")
    parser.add_argument("--personas", type=Path, default=personas_path(), help="Personas JSON file.")
    parser.add_argument("--limit-personas", type=int, help="Only generate for first N personas.")
    parser.add_argument("--provider", default="groq", choices=("groq", "local", "openai"), help="Generation provider.")
    parser.add_argument("--model", default=None, help="Model name for selected provider.")
    parser.add_argument("--local-endpoint", default=DEFAULT_LOCAL_ENDPOINT, help="Local Ollama-compatible endpoint.")
    parser.add_argument(
        "--clean-index-path",
        type=Path,
        default=vector_store_dir() / "rag_index.faiss",
        help="Path to clean FAISS index.",
    )
    parser.add_argument(
        "--clean-metadata-path",
        type=Path,
        default=vector_store_dir() / "rag_metadata.json",
        help="Path to clean metadata JSON.",
    )
    parser.add_argument(
        "--poisoned-store-dir",
        type=Path,
        default=results_dir / "poisoned_vector_store",
        help="Output directory for poisoned index and metadata.",
    )
    parser.add_argument("--target-product", default="Product X", help="Target product to favor in poisoned docs.")
    parser.add_argument("--competitor-product", default="Product Y", help="Competitor to negatively frame.")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL, help="Embedding model for similarity scoring.")
    parser.add_argument(
        "--fast-poison-vectors",
        action="store_true",
        help="Use hash-based vectors for injected docs instead of sentence-transformer embeddings.",
    )
    parser.add_argument("--semantic-threshold", type=float, default=10.0, help="Shift threshold for attack success.")
    parser.add_argument("--output-json", type=Path, default=results_dir / "attack_responses.json", help="JSON output path.")
    parser.add_argument("--output-csv", type=Path, default=results_dir / "attack_responses.csv", help="CSV output path.")
    parser.add_argument("--analysis-csv", type=Path, default=results_dir / "attack_analysis.csv", help="Metrics CSV path.")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls and only produce retrieval/pipeline outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.clean_index_path.exists():
        raise FileNotFoundError(f"Missing clean index file: {args.clean_index_path}")
    if not args.clean_metadata_path.exists():
        raise FileNotFoundError(f"Missing clean metadata file: {args.clean_metadata_path}")

    questions = load_questions(question=args.question, questions_file=args.questions_file)
    personas = resolve_personas(args)

    poison_info = create_poisoned_vector_store(
        clean_index_path=args.clean_index_path,
        clean_metadata_path=args.clean_metadata_path,
        output_dir=args.poisoned_store_dir,
        model_name=args.embedding_model,
        use_hash_embeddings=args.fast_poison_vectors,
        domains=[args.domain] if args.domain else None,
        target_product=args.target_product,
        competitor_product=args.competitor_product,
    )

    embedding_model = None if args.dry_run else load_embedding_model(args.embedding_model)
    attack_types = sorted({chunk.get("attack_type", "unknown") for chunk in poison_info["poisoned_chunks"]})
    model_label = _model_label(args.provider, args.model)
    fast_retrieval = args.dry_run
    clean_metadata_records = read_json(args.clean_metadata_path) if fast_retrieval else None
    poisoned_metadata_records = read_json(poison_info["poisoned_metadata_path"]) if fast_retrieval else None

    records: list[dict] = []
    analysis_rows: list[dict] = []

    for question in questions:
        for persona in personas:
            baseline = generate_response_with_store(
                question=question,
                persona=persona,
                index_path=args.clean_index_path,
                metadata_path=args.clean_metadata_path,
                metadata_records=clean_metadata_records if isinstance(clean_metadata_records, list) else None,
                top_k=args.top_k,
                domain=args.domain,
                provider=args.provider,
                model=args.model,
                local_endpoint=args.local_endpoint,
                dry_run=args.dry_run,
                fast_retrieval=fast_retrieval,
            )
            attacked = generate_response_with_store(
                question=question,
                persona=persona,
                index_path=poison_info["poisoned_index_path"],
                metadata_path=poison_info["poisoned_metadata_path"],
                metadata_records=poisoned_metadata_records if isinstance(poisoned_metadata_records, list) else None,
                top_k=args.top_k,
                domain=args.domain,
                provider=args.provider,
                model=args.model,
                local_endpoint=args.local_endpoint,
                dry_run=args.dry_run,
                fast_retrieval=fast_retrieval,
            )
            metrics = evaluate_response_shift(
                baseline_text=baseline["response"],
                attacked_text=attacked["response"],
                threshold_pct=args.semantic_threshold,
                embedding_model_name=args.embedding_model,
                embedding_model=embedding_model,
            )

            persona_id = str(persona.get("persona_id", "unknown"))
            record = {
                "persona_id": persona_id,
                "persona": persona,
                "prompt": question,
                "question": question,
                "domain": args.domain,
                "provider": args.provider,
                "model": args.model,
                "model_label": model_label,
                "top_k": args.top_k,
                "baseline_prompt": baseline["prompt"],
                "attacked_prompt": attacked["prompt"],
                "baseline_response": baseline["response"],
                "attacked_response": attacked["response"],
                "baseline_retrieved_sources": baseline["retrieved_sources"],
                "attacked_retrieved_sources": attacked["retrieved_sources"],
                "injected_attack_types": attack_types,
                **metrics,
            }
            records.append(record)
            analysis_rows.append(
                {
                    "persona_id": persona_id,
                    "prompt": question,
                    "domain": args.domain,
                    "provider": args.provider,
                    "model": model_label,
                    "response_similarity": metrics["response_similarity"],
                    "semantic_shift_pct": metrics["semantic_shift_pct"],
                    "keyword_overlap": metrics["keyword_overlap"],
                    "keyword_shift_pct": metrics["keyword_shift_pct"],
                    "attack_success": metrics["attack_success"],
                    "threshold_pct": metrics["threshold_pct"],
                }
            )

    write_json(args.output_json, records)
    write_rows_to_csv(
        args.output_csv,
        records,
        fieldnames=[
            "persona_id",
            "prompt",
            "domain",
            "provider",
            "model",
            "model_label",
            "top_k",
            "baseline_response",
            "attacked_response",
            "response_similarity",
            "semantic_shift_pct",
            "keyword_overlap",
            "keyword_shift_pct",
            "attack_success",
            "injected_attack_types",
            "baseline_retrieved_sources",
            "attacked_retrieved_sources",
            "persona",
        ],
    )
    write_rows_to_csv(
        args.analysis_csv,
        analysis_rows,
        fieldnames=[
            "persona_id",
            "prompt",
            "domain",
            "provider",
            "model",
            "response_similarity",
            "semantic_shift_pct",
            "keyword_overlap",
            "keyword_shift_pct",
            "attack_success",
            "threshold_pct",
        ],
    )

    print(f"Poisoned store written to {poison_info['poisoned_index_path'].parent}")
    print(f"Wrote {len(records)} attack comparison records to {args.output_json}")
    print(f"Wrote attack response table to {args.output_csv}")
    print(f"Wrote attack analysis table to {args.analysis_csv}")


if __name__ == "__main__":
    main()
