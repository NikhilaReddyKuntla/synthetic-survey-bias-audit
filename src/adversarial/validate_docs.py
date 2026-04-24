from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.adversarial.generate_attacks import default_attack_docs_path
from src.attacks.poison_utils import tokenize
from src.utils.helpers import outputs_dir, read_json, vector_store_dir, write_json

TRUST_LEVELS = ("low", "medium", "high")
TRUST_ORDER = {label: index for index, label in enumerate(TRUST_LEVELS)}
ABSOLUTE_LANGUAGE_MARKERS = (
    "all ",
    "always ",
    "never ",
    "equal across all",
    "every ",
)
STAT_PATTERN = re.compile(r"\b\d{2,}%\b|\b\d{3,}\s+respondents?\b", re.IGNORECASE)
LOW_ALIGNMENT_THRESHOLD = 0.045
UNSUPPORTED_CLAIM_THRESHOLD = 0.12


def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 1.0
    return float(len(tokens_a & tokens_b) / len(union))


def _group_trusted_chunks_by_domain(trusted_chunks: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for chunk in trusted_chunks:
        if not isinstance(chunk, dict):
            continue
        domain = str(chunk.get("domain") or "general").strip().lower() or "general"
        grouped.setdefault(domain, []).append(chunk)
    return grouped


def _contains_absolute_language(text: str) -> bool:
    lowered = f"{text.lower()} "
    return any(marker in lowered for marker in ABSOLUTE_LANGUAGE_MARKERS)


def _contains_unverified_statistical_claim(text: str) -> bool:
    return bool(STAT_PATTERN.search(text))


def evaluate_attack_document(attack_doc: dict, trusted_chunks_by_domain: dict[str, list[dict]]) -> dict:
    domain = str(attack_doc.get("domain") or "general").strip().lower() or "general"
    target_claim = str(attack_doc.get("target_claim") or "").strip()
    poisoned_text = str(attack_doc.get("poisoned_text") or "").strip()

    doc_tokens = tokenize(f"{target_claim} {poisoned_text}")
    claim_tokens = tokenize(target_claim)
    domain_trusted_chunks = trusted_chunks_by_domain.get(domain, [])
    fallback_chunks = domain_trusted_chunks or [
        chunk for records in trusted_chunks_by_domain.values() for chunk in records
    ]

    chunk_token_sets = [tokenize(str(chunk.get("text") or "")) for chunk in fallback_chunks]
    max_chunk_overlap = max((_jaccard_similarity(doc_tokens, token_set) for token_set in chunk_token_sets), default=0.0)
    claim_support = max((_jaccard_similarity(claim_tokens, token_set) for token_set in chunk_token_sets), default=0.0)

    has_absolute_language = _contains_absolute_language(f"{target_claim} {poisoned_text}")
    has_unverified_stats = _contains_unverified_statistical_claim(poisoned_text)
    unsupported_claim = claim_support < UNSUPPORTED_CLAIM_THRESHOLD

    score = 3
    reasons: list[str] = []
    if unsupported_claim:
        score -= 1
        reasons.append("unsupported_target_claim")
    if max_chunk_overlap < LOW_ALIGNMENT_THRESHOLD:
        score -= 1
        reasons.append("low_alignment_with_trusted_chunks")
    if has_absolute_language:
        score -= 1
        reasons.append("absolute_or_overgeneralized_language")
    if has_unverified_stats and claim_support < 0.2:
        score -= 1
        reasons.append("unverified_statistical_assertion")

    if score <= 1:
        trust_score = "low"
    elif score == 2:
        trust_score = "medium"
    else:
        trust_score = "high"

    if unsupported_claim:
        trust_score = "low"

    recommended_action = "exclude_from_retrieval" if trust_score == "low" else "allow_with_monitoring"

    return {
        "domain": domain,
        "attack_type": attack_doc.get("attack_type"),
        "target_claim": target_claim,
        "poisoned_text": poisoned_text,
        "trust_score": trust_score,
        "max_chunk_overlap": max_chunk_overlap,
        "claim_support": claim_support,
        "has_absolute_language": has_absolute_language,
        "has_unverified_stats": has_unverified_stats,
        "unsupported_claim": unsupported_claim,
        "flagged": trust_score != "high",
        "recommended_action": recommended_action,
        "reasons": reasons,
        "attack_doc": attack_doc,
    }


def validate_attack_documents(attack_docs: list[dict], trusted_chunks: list[dict]) -> dict:
    grouped_trusted = _group_trusted_chunks_by_domain(trusted_chunks)
    evaluations = [evaluate_attack_document(doc, grouped_trusted) for doc in attack_docs if isinstance(doc, dict)]

    trust_counter = Counter(item["trust_score"] for item in evaluations)
    flagged_docs = [item for item in evaluations if item["flagged"]]
    excluded_docs = [item for item in evaluations if item["trust_score"] == "low"]
    kept_docs = [item["attack_doc"] for item in evaluations if item["trust_score"] != "low"]

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


def filter_attack_docs_by_trust(report: dict, minimum_trust: str = "medium") -> list[dict]:
    minimum = minimum_trust.strip().lower()
    if minimum not in TRUST_ORDER:
        raise ValueError(f"Unsupported trust level '{minimum_trust}'. Expected one of {TRUST_LEVELS}.")

    allowed_docs: list[dict] = []
    for item in report.get("validated_documents", []):
        if not isinstance(item, dict):
            continue
        trust_score = str(item.get("trust_score") or "low").lower()
        attack_doc = item.get("attack_doc")
        if trust_score in TRUST_ORDER and TRUST_ORDER[trust_score] >= TRUST_ORDER[minimum]:
            if isinstance(attack_doc, dict):
                allowed_docs.append(attack_doc)
    return allowed_docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate adversarial attack documents with trust heuristics.")
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
        "--output",
        type=Path,
        default=outputs_dir() / "adversarial_validation_report.json",
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

    report = validate_attack_documents(attack_docs=attack_docs, trusted_chunks=trusted_chunks)
    write_json(args.output, report)

    distribution = report["trust_distribution"]
    print(
        "Validation complete:"
        f" high={distribution['high']}, medium={distribution['medium']}, low={distribution['low']}."
    )
    print(f"Wrote validation report to {args.output}")


if __name__ == "__main__":
    main()
