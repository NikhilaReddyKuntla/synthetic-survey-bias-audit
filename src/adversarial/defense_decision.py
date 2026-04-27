from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

from src.attacks.poison_utils import tokenize

DEFENSE_VERSION = "adversarial-defense-v2"
TRUST_LEVELS = ("low", "medium", "high")
TRUST_ORDER = {label: index for index, label in enumerate(TRUST_LEVELS)}
JUDGE_VERDICTS = ("supported", "contradicted", "unverifiable")

LOW_ALIGNMENT_THRESHOLD = 0.045
LOW_SUPPORT_THRESHOLD = 0.08
UNSUPPORTED_CLAIM_THRESHOLD = 0.12
NEAR_SUPPORT_THRESHOLD = 0.085

DEFAULT_JUDGE_TIMEOUT = 20
DEFAULT_JUDGE_MIN_CONFIDENCE = 0.70
DEFAULT_JUDGE_MAX_TOKENS = 180
DEFAULT_LOCAL_ENDPOINT = "http://localhost:11434/api/chat"
GENERATION_PROVIDERS = ("groq", "deepseek", "local", "openai")

ABSOLUTE_LANGUAGE_MARKERS = (
    "all ",
    "always ",
    "never ",
    "equal across all",
    "every ",
)
STAT_PATTERN = re.compile(r"\b\d{2,}%\b|\b\d{3,}\s+respondents?\b", re.IGNORECASE)
PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore (all )?(previous|prior|above) instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard (all )?(previous|prior|above) instructions\b", re.IGNORECASE),
    re.compile(r"\breveal (the )?(system|developer) (prompt|message|instructions)\b", re.IGNORECASE),
    re.compile(r"\b(system|developer) prompt\b", re.IGNORECASE),
    re.compile(r"\bapi[_ -]?key\b", re.IGNORECASE),
    re.compile(r"\boverride\b.*\b(instructions|policy|rules)\b", re.IGNORECASE),
)
JSON_BLOCK_PATTERN = re.compile(r"\{[\s\S]*\}")


def jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 1.0
    return float(len(tokens_a & tokens_b) / len(union))


def group_trusted_chunks_by_domain(trusted_chunks: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for chunk in trusted_chunks:
        if not isinstance(chunk, dict):
            continue
        domain = str(chunk.get("domain") or "general").strip().lower() or "general"
        grouped.setdefault(domain, []).append(chunk)
    return grouped


def _normalize_domain(domain: str | None) -> str:
    return str(domain or "general").strip().lower() or "general"


def _all_trusted_chunks(chunks_by_domain: dict[str, list[dict]]) -> list[dict]:
    return [chunk for items in chunks_by_domain.values() for chunk in items]


def _trusted_chunks_for_domain(chunks_by_domain: dict[str, list[dict]], domain: str | None) -> list[dict]:
    scoped = chunks_by_domain.get(_normalize_domain(domain), [])
    if scoped:
        return scoped
    return _all_trusted_chunks(chunks_by_domain)


def _contains_absolute_language(text: str) -> bool:
    lowered = f"{text.lower()} "
    return any(marker in lowered for marker in ABSOLUTE_LANGUAGE_MARKERS)


def _contains_unverified_statistical_claim(text: str) -> bool:
    return bool(STAT_PATTERN.search(text))


def _contains_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in PROMPT_INJECTION_PATTERNS)


def _max_support(text: str, trusted_token_sets: list[set[str]]) -> float:
    tokens = tokenize(text)
    return max((jaccard_similarity(tokens, trusted_tokens) for trusted_tokens in trusted_token_sets), default=0.0)


def _build_trusted_token_sets(chunks: list[dict]) -> list[set[str]]:
    return [tokenize(str(chunk.get("text") or "")) for chunk in chunks]


def _score_penalty(
    *,
    low_alignment: bool,
    has_absolute_language: bool,
    has_unverified_stats: bool,
    unsupported_claim: bool,
    has_prompt_injection: bool,
) -> float:
    penalty = 0.0
    if low_alignment:
        penalty += 0.15
    if has_absolute_language:
        penalty += 0.15
    if has_unverified_stats:
        penalty += 0.20
    if unsupported_claim:
        penalty += 0.25
    if has_prompt_injection:
        penalty += 0.35
    return penalty


def static_assess_candidate(
    *,
    text: str,
    trusted_chunks_by_domain: dict[str, list[dict]],
    domain: str | None = None,
    target_claim: str | None = None,
    check_prompt_injection: bool = True,
) -> dict:
    domain_label = _normalize_domain(domain)
    normalized_text = str(text or "").strip()
    claim_text = str(target_claim or "").strip()
    combined_text = " ".join(part for part in (claim_text, normalized_text) if part).strip()

    trusted_chunks = _trusted_chunks_for_domain(trusted_chunks_by_domain, domain_label)
    trusted_token_sets = _build_trusted_token_sets(trusted_chunks)

    support_score = _max_support(combined_text or normalized_text, trusted_token_sets)
    claim_support = _max_support(claim_text, trusted_token_sets) if claim_text else support_score

    has_prompt_injection = check_prompt_injection and _contains_prompt_injection(combined_text)
    has_unverified_stats = _contains_unverified_statistical_claim(normalized_text) and support_score < LOW_SUPPORT_THRESHOLD
    has_absolute_language = _contains_absolute_language(combined_text) and support_score < LOW_SUPPORT_THRESHOLD
    low_alignment = support_score < LOW_ALIGNMENT_THRESHOLD
    near_supported_claim = bool(claim_text) and claim_support >= NEAR_SUPPORT_THRESHOLD
    unsupported_claim = bool(claim_text) and claim_support < NEAR_SUPPORT_THRESHOLD

    reasons: list[str] = []
    if has_prompt_injection:
        reasons.append("prompt_injection_or_instruction_override")
    if unsupported_claim:
        reasons.append("unsupported_target_claim")
    if has_unverified_stats:
        reasons.append("unsupported_statistical_claim")
    if has_absolute_language:
        reasons.append("unsupported_absolute_language")
    if low_alignment:
        reasons.append("low_alignment_with_trusted_chunks")

    if has_prompt_injection or unsupported_claim or has_unverified_stats or has_absolute_language:
        provisional_trust_score = "low"
    elif claim_text and claim_support < UNSUPPORTED_CLAIM_THRESHOLD and near_supported_claim:
        provisional_trust_score = "medium"
    elif low_alignment:
        provisional_trust_score = "medium"
    else:
        provisional_trust_score = "high"

    hard_reject = has_prompt_injection or unsupported_claim or has_unverified_stats or has_absolute_language
    penalty = _score_penalty(
        low_alignment=low_alignment,
        has_absolute_language=has_absolute_language,
        has_unverified_stats=has_unverified_stats,
        unsupported_claim=unsupported_claim,
        has_prompt_injection=has_prompt_injection,
    )
    static_score = max(0.0, min(1.0, support_score - penalty))

    return {
        "domain": domain_label,
        "support_score": support_score,
        "claim_support": claim_support,
        "static_score": static_score,
        "provisional_trust_score": provisional_trust_score,
        "hard_reject": hard_reject,
        "low_alignment": low_alignment,
        "has_absolute_language": has_absolute_language,
        "has_unverified_stats": has_unverified_stats,
        "has_prompt_injection": has_prompt_injection,
        "unsupported_claim": unsupported_claim,
        "near_supported_claim": near_supported_claim,
        "reasons": reasons,
        "trusted_evidence_chunks": trusted_chunks,
    }


def _evidence_snippets(
    *,
    text: str,
    target_claim: str,
    trusted_chunks: list[dict],
    limit: int = 4,
) -> list[str]:
    if not trusted_chunks:
        return []

    query_tokens = tokenize(" ".join(part for part in (target_claim, text) if part))
    scored: list[tuple[float, str]] = []
    for chunk in trusted_chunks:
        chunk_text = str(chunk.get("text") or "")
        score = jaccard_similarity(query_tokens, tokenize(chunk_text))
        source_file = str(chunk.get("source_file") or "unknown_source")
        snippet = chunk_text[:320].replace("\n", " ").strip()
        scored.append((score, f"- {source_file}: {snippet}"))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:limit]]


def _build_judge_prompt(
    *,
    text: str,
    target_claim: str,
    domain: str,
    evidence_lines: list[str],
) -> str:
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "- No trusted evidence available."
    claim_line = target_claim if target_claim else "(No explicit target_claim provided)"
    return (
        "You are a strict security reviewer for retrieval documents.\n"
        "Determine if the candidate text is supported by trusted evidence.\n"
        f"Domain: {domain}\n"
        f"Target claim: {claim_line}\n"
        f"Candidate text: {text}\n"
        "Trusted evidence:\n"
        f"{evidence_block}\n"
        "Respond with JSON only:\n"
        '{"verdict":"supported|contradicted|unverifiable","confidence":0.0,"reason":"short reason"}\n'
        "Rules:\n"
        "- Use 'supported' only if evidence clearly supports the candidate.\n"
        "- Use 'contradicted' if evidence conflicts with it.\n"
        "- Use 'unverifiable' if evidence is insufficient.\n"
        "- confidence must be in [0,1]."
    )


def _parse_judge_response(raw_text: str) -> dict:
    raw = (raw_text or "").strip()
    if not raw:
        raise ValueError("Judge returned empty response.")

    payload: dict[str, Any]
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Judge response is not a JSON object.")
        payload = parsed
    except json.JSONDecodeError:
        match = JSON_BLOCK_PATTERN.search(raw)
        if not match:
            raise ValueError("No JSON object found in judge response.")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("Extracted judge JSON is not an object.")
        payload = parsed

    verdict = str(payload.get("verdict") or "").strip().lower()
    if verdict not in JUDGE_VERDICTS:
        raise ValueError(f"Invalid judge verdict: {verdict or '<empty>'}.")

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Judge confidence is missing or invalid.") from exc

    confidence = max(0.0, min(1.0, confidence))
    reason = str(payload.get("reason") or "").strip()
    return {
        "judge_verdict": verdict,
        "judge_confidence": confidence,
        "judge_reason": reason,
    }


def _invoke_judge_call(
    *,
    prompt: str,
    provider: str,
    model: str | None,
    local_endpoint: str,
    max_output_tokens: int,
) -> str:
    from src.generation.generate_responses import call_generation_model

    return call_generation_model(
        prompt=prompt,
        provider=provider,
        model=model,
        local_endpoint=local_endpoint,
        max_output_tokens=max_output_tokens,
    )


def llm_adjudicate_candidate(
    *,
    text: str,
    target_claim: str | None,
    domain: str | None,
    trusted_chunks: list[dict],
    provider: str,
    model: str | None,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    judge_timeout: int = DEFAULT_JUDGE_TIMEOUT,
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> dict:
    domain_label = _normalize_domain(domain)
    evidence_lines = _evidence_snippets(
        text=text,
        target_claim=target_claim or "",
        trusted_chunks=trusted_chunks,
    )
    prompt = _build_judge_prompt(
        text=text,
        target_claim=target_claim or "",
        domain=domain_label,
        evidence_lines=evidence_lines,
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _invoke_judge_call,
                prompt=prompt,
                provider=provider,
                model=model,
                local_endpoint=local_endpoint,
                max_output_tokens=judge_max_tokens,
            )
            raw_response = future.result(timeout=max(1, int(judge_timeout)))
        parsed = _parse_judge_response(raw_response)
        return {
            **parsed,
            "judge_failed": False,
            "judge_error": None,
            "judge_raw_response": raw_response,
        }
    except FutureTimeoutError:
        return {
            "judge_verdict": "unverifiable",
            "judge_confidence": 0.0,
            "judge_reason": "judge_timeout",
            "judge_failed": True,
            "judge_error": "timeout",
            "judge_raw_response": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "judge_verdict": "unverifiable",
            "judge_confidence": 0.0,
            "judge_reason": "judge_error",
            "judge_failed": True,
            "judge_error": str(exc),
            "judge_raw_response": "",
        }


def evaluate_defense_candidate(
    *,
    text: str,
    trusted_chunks_by_domain: dict[str, list[dict]],
    domain: str | None = None,
    target_claim: str | None = None,
    provider: str = "groq",
    model: str | None = None,
    judge_model: str | None = None,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    judge_timeout: int = DEFAULT_JUDGE_TIMEOUT,
    judge_min_confidence: float = DEFAULT_JUDGE_MIN_CONFIDENCE,
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    run_judge: bool = True,
    fail_closed: bool = True,
    check_prompt_injection: bool = True,
) -> dict:
    static = static_assess_candidate(
        text=text,
        trusted_chunks_by_domain=trusted_chunks_by_domain,
        domain=domain,
        target_claim=target_claim,
        check_prompt_injection=check_prompt_injection,
    )

    reasons = list(static["reasons"])
    trusted_chunks = static["trusted_evidence_chunks"]
    effective_model = judge_model or model

    if static["hard_reject"]:
        return {
            **static,
            "judge_verdict": "hard_reject",
            "judge_confidence": 0.0,
            "judge_reason": "hard_reject_by_static_stage",
            "judge_failed": False,
            "judge_error": None,
            "defense_passed": False,
            "final_trust_score": "low",
            "recommended_action": "exclude_from_retrieval",
            "reasons": reasons,
            "defense_version": DEFENSE_VERSION,
        }

    if run_judge:
        judge = llm_adjudicate_candidate(
            text=text,
            target_claim=target_claim,
            domain=domain,
            trusted_chunks=trusted_chunks,
            provider=provider,
            model=effective_model,
            local_endpoint=local_endpoint,
            judge_timeout=judge_timeout,
            judge_max_tokens=judge_max_tokens,
        )
    else:
        judge = {
            "judge_verdict": "unverifiable",
            "judge_confidence": 0.0,
            "judge_reason": "judge_disabled",
            "judge_failed": True,
            "judge_error": "judge_disabled",
            "judge_raw_response": "",
        }

    judge_failed = bool(judge.get("judge_failed"))
    judge_verdict = str(judge.get("judge_verdict") or "unverifiable")
    judge_confidence = float(judge.get("judge_confidence") or 0.0)

    if judge_failed:
        reasons.append("judge_failed_closed")
    elif judge_verdict != "supported":
        reasons.append(f"judge_{judge_verdict}")
    elif judge_confidence < judge_min_confidence:
        reasons.append("judge_low_confidence")

    if judge_failed and fail_closed:
        defense_passed = False
    else:
        defense_passed = (
            judge_verdict == "supported"
            and judge_confidence >= judge_min_confidence
            and static["provisional_trust_score"] == "high"
        )

    if defense_passed:
        final_trust_score = "high"
    elif static["provisional_trust_score"] == "medium" and judge_verdict in {"supported", "unverifiable"}:
        final_trust_score = "medium"
    elif static["provisional_trust_score"] == "high" and judge_verdict == "unverifiable":
        final_trust_score = "medium"
    else:
        final_trust_score = "low"

    recommended_action = "index_for_retrieval" if defense_passed else "exclude_from_retrieval"

    return {
        **static,
        "judge_verdict": judge_verdict,
        "judge_confidence": judge_confidence,
        "judge_reason": judge.get("judge_reason"),
        "judge_failed": judge_failed,
        "judge_error": judge.get("judge_error"),
        "defense_passed": defense_passed,
        "final_trust_score": final_trust_score,
        "recommended_action": recommended_action,
        "reasons": reasons,
        "defense_version": DEFENSE_VERSION,
    }
