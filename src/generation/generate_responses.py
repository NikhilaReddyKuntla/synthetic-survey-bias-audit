from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.attacks.poison_utils import build_adversarial_templates, tokenize
from src.rag.embed import DEFAULT_MODEL
from src.rag.retrieve import format_retrieved_context, retrieve_chunks
from src.utils.helpers import personas_path, read_json, read_jsonl, synthetic_responses_path, write_json
from src.utils.prompt_templates import build_survey_response_prompt

DEFAULT_LOCAL_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_LOCAL_MODEL = "llama3.1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
GENERATION_PROVIDERS = ("groq", "local", "openai")
ALLOWED_DOMAINS = ("ecommerce", "finance", "healthcare")


def load_persona(persona_json: str | None = None, persona_file: Path | None = None) -> dict | None:
    if persona_json:
        return json.loads(persona_json)

    if persona_file:
        records = read_jsonl(persona_file)
        return records[0] if records else None

    return None


def load_personas(path: Path) -> list[dict]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected persona file to contain a JSON list: {path}")
    return [persona for persona in payload if isinstance(persona, dict)]


def load_questions(question: list[str] | None = None, questions_file: Path | None = None) -> list[str]:
    questions: list[str] = []
    if question:
        questions.extend(item for item in question if item.strip())

    if questions_file:
        if questions_file.suffix.lower() == ".json":
            payload = read_json(questions_file)
            if isinstance(payload, list):
                questions.extend(str(item) for item in payload)
            else:
                raise ValueError(f"Expected JSON questions file to contain a list: {questions_file}")
        else:
            with questions_file.open("r", encoding="utf-8") as handle:
                questions.extend(line.strip() for line in handle if line.strip())

    if not questions:
        raise ValueError("Provide at least one survey question with --question or --questions-file.")

    return questions


def source_refs(chunks: list[dict]) -> list[dict]:
    return [
        {
            "chunk_id": chunk.get("chunk_id"),
            "chunk_type": chunk.get("chunk_type", "text"),
            "domain": chunk.get("domain"),
            "source_file": chunk.get("source_file"),
            "year": chunk.get("year"),
            "page_number": chunk.get("page_number"),
        }
        for chunk in chunks
    ]


def infer_user_doc_domain(domain: str | None, purpose: str | None) -> str:
    if domain:
        return domain
    purpose_text = (purpose or "").lower()
    for candidate in ALLOWED_DOMAINS:
        if candidate in purpose_text:
            return candidate
    return "finance"


def score_chunks_lexical(query: str, chunks: list[dict]) -> list[dict]:
    query_tokens = tokenize(query or "")
    scored: list[tuple[int, int, dict]] = []
    for position, chunk in enumerate(chunks):
        text_tokens = tokenize(str(chunk.get("text") or ""))
        scored.append((len(query_tokens & text_tokens), -position, chunk))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return [chunk for _, _, chunk in scored]


def validate_user_doc_chunks_lexical(chunks: list[dict], threshold: float) -> tuple[bool, list[str]]:
    suspicious_phrases = (
        "ignore previous",
        "ignore all previous",
        "system prompt",
        "developer message",
        "override instructions",
        "you must answer",
        "do not follow",
    )
    reasons: list[str] = []
    combined_text = "\n".join(str(chunk.get("text") or "") for chunk in chunks)
    lowered_text = combined_text.lower()
    for phrase in suspicious_phrases:
        if phrase in lowered_text:
            reasons.append(f"prompt_injection_phrase:{phrase}")

    uploaded_tokens = tokenize(combined_text)
    template_texts = [
        template["text"]
        for domain in ALLOWED_DOMAINS
        for template in build_adversarial_templates(domain)
    ]
    for template in template_texts:
        template_tokens = tokenize(template)
        if not uploaded_tokens or not template_tokens:
            continue
        overlap = len(uploaded_tokens & template_tokens) / len(template_tokens)
        if overlap >= threshold:
            reasons.append(f"adversarial_template_overlap:{overlap:.2f}")
            break

    return bool(reasons), reasons


def prepare_user_doc_chunks(
    user_doc: Path,
    domain: str,
    purpose: str | None,
    adversarial_threshold: float,
    embedding_model_name: str = DEFAULT_MODEL,
) -> tuple[list[dict], dict]:
    from src.pipeline.document_pipeline import build_upload_chunks, parse_document

    text_chunks = parse_document(user_doc)
    if not text_chunks:
        raise ValueError(f"User document produced no text after parsing: {user_doc}")

    chunks = build_upload_chunks(text_chunks=text_chunks, doc_path=user_doc, domain=domain)
    for chunk in chunks:
        chunk["source_kind"] = "user_upload"
        chunk["upload_purpose"] = purpose

    is_flagged, reasons = validate_user_doc_chunks_lexical(chunks=chunks, threshold=adversarial_threshold)
    validation = {
        "user_doc": str(user_doc),
        "user_doc_purpose": purpose,
        "domain": domain,
        "chunks": len(chunks),
        "adversarial_threshold": adversarial_threshold,
        "defense_passed": not is_flagged,
        "reasons": reasons,
    }
    if is_flagged:
        raise ValueError(
            "User document failed adversarial validation and was not used. "
            f"Reasons: {reasons}"
        )
    return chunks, validation


def normalize_user_doc_path(path: Path) -> Path:
    if path.exists():
        return path
    normalized = Path(str(path).replace("\\", "/"))
    return normalized if normalized.exists() else path


def generate_with_groq(prompt: str, model: str = DEFAULT_GROQ_MODEL, max_output_tokens: int = 500) -> str:
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("Groq generation requires `groq`. Install with `pip install -r requirements.txt`.") from exc

    client = Groq()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=max_output_tokens,
    )
    return (completion.choices[0].message.content or "").strip()


def generate_with_openai(prompt: str, model: str = DEFAULT_OPENAI_MODEL, max_output_tokens: int = 500) -> str:
    import os
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_output_tokens,
    )
    return response.choices[0].message.content.strip()


def generate_with_local(
    prompt: str,
    model: str = DEFAULT_LOCAL_MODEL,
    endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    max_output_tokens: int = 500,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": max_output_tokens},
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(
            f"Unable to reach local generation endpoint at {endpoint}. "
            "Start your local server first, for example `ollama serve`."
        ) from exc

    return (response_payload.get("message", {}).get("content") or "").strip()


def call_generation_model(
    prompt: str,
    provider: str = "groq",
    model: str | None = None,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    max_output_tokens: int = 500,
) -> str:
    if provider == "groq":
        return generate_with_groq(prompt, model=model or DEFAULT_GROQ_MODEL, max_output_tokens=max_output_tokens)
    if provider == "local":
        return generate_with_local(
            prompt,
            model=model or DEFAULT_LOCAL_MODEL,
            endpoint=local_endpoint,
            max_output_tokens=max_output_tokens,
        )
    if provider == "openai":
        return generate_with_openai(prompt, model=model or DEFAULT_OPENAI_MODEL, max_output_tokens=max_output_tokens)
    raise ValueError("Unsupported generation provider. Expected 'groq', 'local', or 'openai'.")


def generate_response(
    question: str,
    domain: str | None = None,
    top_k: int = 3,
    persona: dict | None = None,
    provider: str = "groq",
    model: str | None = None,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    user_doc_chunks: list[dict] | None = None,
    user_doc_validation: dict | None = None,
    dry_run: bool = False,
) -> dict:
    if user_doc_chunks:
        prioritized_user_chunks = score_chunks_lexical(question, user_doc_chunks)
        clean_chunks = retrieve_chunks(query=question, top_k=top_k, domain=domain)
        used_ids: set[str] = set()
        chunks = []
        for chunk in prioritized_user_chunks + clean_chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id in used_ids:
                continue
            chunks.append(chunk)
            used_ids.add(chunk_id)
            if len(chunks) >= top_k:
                break
    else:
        chunks = retrieve_chunks(query=question, top_k=top_k, domain=domain)
    retrieved_context = format_retrieved_context(chunks)
    prompt = build_survey_response_prompt(
        question=question,
        retrieved_context=retrieved_context,
        persona=persona,
    )

    response_text = "" if dry_run else call_generation_model(
        prompt=prompt,
        provider=provider,
        model=model,
        local_endpoint=local_endpoint,
    )

    return {
        "persona_id": persona.get("persona_id") if persona else None,
        "question": question,
        "domain": domain,
        "persona": persona,
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "retrieved_sources": source_refs(chunks),
        "user_doc_validation": user_doc_validation,
        "prompt": prompt,
        "response": response_text,
        "dry_run": dry_run,
    }


def generate_responses(
    questions: list[str],
    personas: list[dict],
    domain: str | None = None,
    top_k: int = 3,
    provider: str = "groq",
    model: str | None = None,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    user_doc_chunks: list[dict] | None = None,
    user_doc_validation: dict | None = None,
    dry_run: bool = False,
) -> list[dict]:
    records: list[dict] = []
    for question in questions:
        for persona in personas:
            records.append(
                generate_response(
                    question=question,
                    domain=domain,
                    top_k=top_k,
                    persona=persona,
                    provider=provider,
                    model=model,
                    local_endpoint=local_endpoint,
                    user_doc_chunks=user_doc_chunks,
                    user_doc_validation=user_doc_validation,
                    dry_run=dry_run,
                )
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic survey response using RAG context.")
    parser.add_argument("--question", action="append", help="Survey question to answer. Can be provided multiple times.")
    parser.add_argument("--questions-file", type=Path, help="Text file with one question per line, or JSON list.")
    parser.add_argument("--domain", choices=("ecommerce", "finance", "healthcare"), help="Optional RAG domain filter.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of RAG chunks to retrieve.")
    parser.add_argument("--persona-json", help="Optional persona as a JSON object string.")
    parser.add_argument("--persona-file", type=Path, help="Optional JSONL persona file. Uses the first record.")
    parser.add_argument("--personas", type=Path, default=personas_path(), help="Personas JSON file.")
    parser.add_argument("--limit-personas", type=int, help="Only generate for the first N personas.")
    parser.add_argument("--provider", default="groq", choices=GENERATION_PROVIDERS, help="Generation provider.")
    parser.add_argument("--model", default=None, help="Generation model. Defaults depend on provider.")
    parser.add_argument("--local-endpoint", default=DEFAULT_LOCAL_ENDPOINT, help="Local Ollama-compatible chat endpoint.")
    parser.add_argument("--user-doc", type=Path, help="Optional user document (.pdf, .txt, .md) to validate and prioritize.")
    parser.add_argument("--user-doc-purpose", help="Purpose label for the user document.")
    parser.add_argument(
        "--adversarial-threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for user-doc adversarial detection.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL,
        help="Sentence-Transformers model used for user-doc validation.",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=None,
        help="Compatibility flag accepted for older docs; current user-doc validation is heuristic.",
    )
    parser.add_argument(
        "--judge-min-confidence",
        type=float,
        default=None,
        help="Compatibility flag accepted for older docs; current user-doc validation is heuristic.",
    )
    parser.add_argument("--output", type=Path, default=synthetic_responses_path(), help="JSON output path.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompt and retrieve context without calling an LLM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(question=args.question, questions_file=args.questions_file)
    if args.persona_json or args.persona_file:
        persona = load_persona(persona_json=args.persona_json, persona_file=args.persona_file)
        personas = [persona] if persona else []
    else:
        personas = load_personas(args.personas)

    if args.limit_personas is not None:
        personas = personas[: args.limit_personas]
    if not personas:
        raise ValueError("No personas available for generation.")

    user_doc_chunks = None
    user_doc_validation = None
    generation_domain = args.domain
    if args.user_doc:
        args.user_doc = normalize_user_doc_path(args.user_doc)
        if not args.user_doc.exists():
            raise FileNotFoundError(f"User document does not exist: {args.user_doc}")
        user_doc_domain = infer_user_doc_domain(args.domain, args.user_doc_purpose)
        generation_domain = generation_domain or user_doc_domain
        user_doc_chunks, user_doc_validation = prepare_user_doc_chunks(
            user_doc=args.user_doc,
            domain=user_doc_domain,
            purpose=args.user_doc_purpose,
            adversarial_threshold=args.adversarial_threshold,
            embedding_model_name=args.embedding_model,
        )

    records = generate_responses(
        questions=questions,
        personas=personas,
        domain=generation_domain,
        top_k=args.top_k,
        provider=args.provider,
        model=args.model,
        local_endpoint=args.local_endpoint,
        user_doc_chunks=user_doc_chunks,
        user_doc_validation=user_doc_validation,
        dry_run=args.dry_run,
    )
    write_json(args.output, records)

    if args.dry_run:
        for record in records[:3]:
            print(record["prompt"])
            print("\n---\n")
    else:
        print(f"Generated {len(records)} synthetic responses.")
    print(f"Wrote generation records to {args.output}")


if __name__ == "__main__":
    main()
