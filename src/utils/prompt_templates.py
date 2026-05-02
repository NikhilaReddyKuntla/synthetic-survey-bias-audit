from __future__ import annotations

import json
import re

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
PHRASE_SPLIT_PATTERN = re.compile(r",|;|\band\b|\bor\b", re.IGNORECASE)
STOPWORDS = {
    "about",
    "affect",
    "customer",
    "customers",
    "does",
    "from",
    "how",
    "product",
    "specific",
    "survey",
    "the",
    "this",
    "what",
    "which",
    "would",
}


def format_persona(persona: dict | None = None) -> str:
    if not persona:
        return "No specific persona was provided. Answer as a realistic survey respondent."

    profile_text = persona.get("profile_text")
    if profile_text:
        return str(profile_text)

    return json.dumps(persona, ensure_ascii=False, indent=2)


def _normalize_phrase(phrase: str) -> str:
    words = [word.lower() for word in TOKEN_PATTERN.findall(phrase)]
    words = [word for word in words if len(word) > 2 and word not in STOPWORDS]
    return " ".join(words).strip()


def extract_grounding_phrases(
    question: str,
    retrieved_context: str,
    max_phrases: int = 8,
) -> list[str]:
    """Pull concise factor phrases that the answer should reuse when natural."""
    context_tokens = set(TOKEN_PATTERN.findall(retrieved_context.lower()))
    phrases: list[str] = []
    seen: set[str] = set()

    question_tail = re.sub(r"^for\s+product\s+x\s+\w+\s*,\s*how\s+do\s+", "", question, flags=re.IGNORECASE)
    question_tail = re.sub(r"\baffect\b.*$", "", question_tail, flags=re.IGNORECASE)
    for raw_phrase in PHRASE_SPLIT_PATTERN.split(question_tail):
        phrase = _normalize_phrase(raw_phrase)
        if not phrase or phrase in seen:
            continue
        phrase_tokens = phrase.split()
        if not phrase_tokens or len(phrase_tokens) > 5:
            continue
        if all(token in context_tokens for token in phrase_tokens):
            phrases.append(phrase)
            seen.add(phrase)
        if len(phrases) >= max_phrases:
            return phrases

    return phrases


def build_survey_response_prompt(
    question: str,
    retrieved_context: str,
    persona: dict | None = None,
) -> str:
    persona_block = format_persona(persona)
    grounding_phrases = extract_grounding_phrases(question, retrieved_context)
    grounding_line = (
        "When natural, reuse these exact grounding phrases from the question/context: "
        + ", ".join(grounding_phrases)
        if grounding_phrases
        else "Use the retrieved context's wording for key factors when natural."
    )
    return f"""
Answer specifically with respect to this survey question:
{question}

You are answering as the following respondent:
{persona_block}

Use the following retrieved background context where it is relevant:
{retrieved_context}

Answer from the respondent's perspective.
Use only the provided context where it is relevant.
If the context does not support a specific claim, answer generally from the persona's perspective.
{grounding_line}
Mention 3-6 concrete context factors rather than broad synonyms.
Do not copy the context directly.
Do not invent precise statistics unless they appear in the retrieved context.
Keep the response short: 1-2 survey-style sentences.
Return only the respondent's answer.
""".strip()
