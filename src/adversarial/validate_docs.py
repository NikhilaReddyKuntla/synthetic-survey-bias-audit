from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.adversarial.defense_decision import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MIN_CONFIDENCE,
    DEFAULT_JUDGE_TIMEOUT,
    DEFAULT_LOCAL_ENDPOINT,
    GENERATION_PROVIDERS,
    TRUST_LEVELS,
    TRUST_ORDER,
    evaluate_defense_candidate,
    group_trusted_chunks_by_domain,
)
from src.adversarial.generate_attacks import default_attack_docs_path
from src.utils.helpers import attack_outputs_dir, read_json, vector_store_dir, write_json


def evaluate_attack_document(
    attack_doc: dict,
    trusted_chunks_by_domain: dict[str, list[dict]],
    provider: str = "groq",
    model: str | None = None,
    judge_model: str | None = None,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    judge_timeout: int = DEFAULT_JUDGE_TIMEOUT,
    judge_min_confidence: float = DEFAULT_JUDGE_MIN_CONFIDENCE,
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> dict:
    domain = str(attack_doc.get("domain") or "general").strip().lower() or "general"
    attack_type = attack_doc.get("attack_type")
    intended_trust_score = attack_doc.get("intended_trust_score")
    target_claim = str(attack_doc.get("target_claim") or "").strip()
    poisoned_text = str(attack_doc.get("poisoned_text") or "").strip()

    decision = evaluate_defense_candidate(
        text=poisoned_text,
        trusted_chunks_by_domain=trusted_chunks_by_domain,
        domain=domain,
        target_claim=target_claim,
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

    final_trust = decision["final_trust_score"]
    return {
        "domain": domain,
        "attack_type": attack_type,
        "intended_trust_score": intended_trust_score,
        "target_claim": target_claim,
        "poisoned_text": poisoned_text,
        "trust_score": final_trust,  # compatibility alias
        "final_trust_score": final_trust,
        "defense_passed": decision["defense_passed"],
        "static_score": decision["static_score"],
        "support_score": decision["support_score"],
        "claim_support": decision["claim_support"],
        "has_absolute_language": decision["has_absolute_language"],
        "has_unverified_stats": decision["has_unverified_stats"],
        "has_prompt_injection": decision["has_prompt_injection"],
        "unsupported_claim": decision["unsupported_claim"],
        "near_supported_claim": decision.get("near_supported_claim"),
        "low_alignment": decision["low_alignment"],
        "judge_verdict": decision["judge_verdict"],
        "judge_confidence": decision["judge_confidence"],
        "judge_reason": decision["judge_reason"],
        "judge_failed": decision["judge_failed"],
        "judge_error": decision["judge_error"],
        "flagged": not decision["defense_passed"],
        "recommended_action": decision["recommended_action"],
        "reasons": decision["reasons"],
        "defense_version": decision["defense_version"],
        "attack_doc": attack_doc,
    }


def validate_attack_documents(
    attack_docs: list[dict],
    trusted_chunks: list[dict],
    provider: str = "groq",
    model: str | None = None,
    judge_model: str | None = None,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    judge_timeout: int = DEFAULT_JUDGE_TIMEOUT,
    judge_min_confidence: float = DEFAULT_JUDGE_MIN_CONFIDENCE,
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> dict:
    grouped_trusted = group_trusted_chunks_by_domain(trusted_chunks)
    evaluations = [
        evaluate_attack_document(
            doc,
            grouped_trusted,
            provider=provider,
            model=model,
            judge_model=judge_model,
            local_endpoint=local_endpoint,
            judge_timeout=judge_timeout,
            judge_min_confidence=judge_min_confidence,
            judge_max_tokens=judge_max_tokens,
        )
        for doc in attack_docs
        if isinstance(doc, dict)
    ]

    trust_counter = Counter(item["final_trust_score"] for item in evaluations)
    flagged_docs = [item for item in evaluations if item["flagged"]]
    excluded_docs = [item for item in evaluations if not item["defense_passed"]]
    kept_docs = [item["attack_doc"] for item in evaluations if item["defense_passed"]]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_documents": len(evaluations),
        "trust_distribution": {
            "high": trust_counter.get("high", 0),
            "medium": trust_counter.get("medium", 0),
            "low": trust_counter.get("low", 0),
        },
        "flagged_chunks": flagged_docs,
        "excluded_low_trust_chunks": excluded_docs,
        "kept_attack_documents": kept_docs,
        "validated_documents": evaluations,
    }


def filter_attack_docs_by_trust(report: dict, minimum_trust: str = "high") -> list[dict]:
    minimum = minimum_trust.strip().lower()
    if minimum not in TRUST_ORDER:
        raise ValueError(f"Unsupported trust level '{minimum_trust}'. Expected one of {TRUST_LEVELS}.")

    allowed_docs: list[dict] = []
    for item in report.get("validated_documents", []):
        if not isinstance(item, dict):
            continue
        trust_score = str(item.get("final_trust_score") or item.get("trust_score") or "low").lower()
        defense_passed = bool(item.get("defense_passed"))
        attack_doc = item.get("attack_doc")
        if defense_passed and trust_score in TRUST_ORDER and TRUST_ORDER[trust_score] >= TRUST_ORDER[minimum]:
            if isinstance(attack_doc, dict):
                allowed_docs.append(attack_doc)
    return allowed_docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate adversarial attack documents with strict defense checks.")
    parser.add_argument(
        "--attack-docs",
        type=Path,
        default=default_attack_docs_path(),
        help="Path to generated attack documents.",
    )
    parser.add_argument(
        "--trusted-metadata-path",
        type=Path,
        default=vector_store_dir() / "rag_metadata.json",
        help="Trusted metadata JSON path for clean RAG chunks.",
    )
    parser.add_argument(
        "--provider",
        default="groq",
        choices=GENERATION_PROVIDERS,
        help="Provider used for LLM adjudication.",
    )
    parser.add_argument("--model", default=None, help="Generation model used by provider unless --judge-model is set.")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional dedicated model for defense adjudication.",
    )
    parser.add_argument("--local-endpoint", default=DEFAULT_LOCAL_ENDPOINT, help="Local provider endpoint.")
    parser.add_argument("--judge-timeout", type=int, default=DEFAULT_JUDGE_TIMEOUT, help="Judge timeout in seconds.")
    parser.add_argument(
        "--judge-min-confidence",
        type=float,
        default=DEFAULT_JUDGE_MIN_CONFIDENCE,
        help="Minimum judge confidence required for acceptance.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_TOKENS,
        help="Maximum completion tokens for judge response.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=attack_outputs_dir() / "adversarial_validation_report.json",
        help="Validation report output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.attack_docs.exists():
        raise FileNotFoundError(f"Attack docs file not found: {args.attack_docs}")
    if not args.trusted_metadata_path.exists():
        raise FileNotFoundError(f"Trusted metadata file not found: {args.trusted_metadata_path}")

    attack_docs = read_json(args.attack_docs)
    trusted_chunks = read_json(args.trusted_metadata_path)
    if not isinstance(attack_docs, list):
        raise ValueError(f"Expected attack docs JSON list in {args.attack_docs}")
    if not isinstance(trusted_chunks, list):
        raise ValueError(f"Expected trusted metadata JSON list in {args.trusted_metadata_path}")

    report = validate_attack_documents(
        attack_docs=attack_docs,
        trusted_chunks=trusted_chunks,
        provider=args.provider,
        model=args.model,
        judge_model=args.judge_model,
        local_endpoint=args.local_endpoint,
        judge_timeout=args.judge_timeout,
        judge_min_confidence=args.judge_min_confidence,
        judge_max_tokens=args.judge_max_tokens,
    )
    write_json(args.output, report)

    distribution = report["trust_distribution"]
    print(
        "Validation complete:"
        f" high={distribution['high']}, medium={distribution['medium']}, low={distribution['low']}."
    )
    print(f"Wrote validation report to {args.output}")


if __name__ == "__main__":
    main()
