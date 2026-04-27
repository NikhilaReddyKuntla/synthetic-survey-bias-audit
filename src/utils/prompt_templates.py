from __future__ import annotations

import json


def format_persona(persona: dict | None = None) -> str:
    if not persona:
        return "No specific persona was provided. Answer as a realistic survey respondent."

    profile_text = persona.get("profile_text")
    if profile_text:
        return str(profile_text)

    return json.dumps(persona, ensure_ascii=False, indent=2)


def build_survey_response_prompt(
    question: str,
    retrieved_context: str,
    persona: dict | None = None,
) -> str:
    persona_block = format_persona(persona)
    return f"""
You are answering as the following respondent:
{persona_block}

Use the following retrieved background context where it is relevant:
{retrieved_context}

Survey question:
{question}

Answer from the respondent's perspective.
Use only the provided context where it is relevant.
If the context does not support a specific claim, answer generally from the persona's perspective.
Do not copy the context directly.
Do not invent precise statistics unless they appear in the retrieved context.
Keep the response short: 1-2 survey-style sentences.
Return only the respondent's answer.
""".strip()
