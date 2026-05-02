from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from src.attacks.poison_utils import build_adversarial_templates, tokenize
from src.rag.embed import DEFAULT_MODEL
from src.rag.ingest import slugify
from src.rag.retrieve import format_retrieved_context, retrieve_chunks
from src.utils.doc_utils import extract_clean_document_text
from src.utils.helpers import personas_path, read_json, read_jsonl, synthetic_responses_path, write_json
from src.utils.prompt_templates import build_survey_response_prompt
from src.utils.text_utils import chunk_text

DEFAULT_LOCAL_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_LOCAL_MODEL = "llama3.1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_GENERATION_TEMPERATURE = 0.35
GENERATION_PROVIDERS = ("groq", "local", "openai")
ALLOWED_DOMAINS = ("ecommerce", "finance", "healthcare")

load_dotenv(override=False)


def infer_domain_from_question(question: str, domain: str | None = None) -> str | None:
    if domain:
        return domain
    lowered = (question or "").lower()
    for candidate in ALLOWED_DOMAINS:
        if candidate in lowered:
            return candidate
    finance_terms = ("budget", "price", "cost", "financial", "finance", "security")
    healthcare_terms = ("health", "healthcare", "patient", "medical", "care")
    ecommerce_terms = ("ecommerce", "shopping", "purchase", "delivery", "return")
    if any(term in lowered for term in finance_terms):
        return "finance"
    if any(term in lowered for term in healthcare_terms):
        return "healthcare"
    if any(term in lowered for term in ecommerce_terms):
        return "ecommerce"
    return None


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
            "similarity_score": chunk.get("similarity_score"),
            "lexical_score": chunk.get("lexical_score"),
            "hybrid_score": chunk.get("hybrid_score"),
        }
        for chunk in chunks
    ]


def infer_user_doc_domain(domain: str | None, purpose: str | None, fallback: str = "finance") -> str:
    if domain:
        return domain
    purpose_text = (purpose or "").lower()
    for candidate in ALLOWED_DOMAINS:
        if candidate in purpose_text:
            return candidate
    return fallback


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
    fail_on_flagged: bool = True,
) -> tuple[list[dict], dict]:
    text = extract_clean_document_text(user_doc)
    if not text:
        raise ValueError(f"User document produced no text after parsing: {user_doc}")

    source_slug = slugify(user_doc.stem)
    chunks = [
        {
            "chunk_id": f"user_doc_{source_slug}_text_{index:03d}",
            "chunk_type": "text",
            "domain": domain,
            "source_file": user_doc.name,
            "source_kind": "user_upload",
            "upload_purpose": purpose,
            "text": chunk,
        }
        for index, chunk in enumerate(chunk_text(text, chunk_size=450, chunk_overlap=80), start=1)
    ]
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
    if is_flagged and fail_on_flagged:
        raise ValueError(
            "User document failed adversarial validation and was not used. "
            f"Reasons: {reasons}"
        )
    return chunks, validation


def prepare_user_docs_chunks(
    user_docs: list[Path],
    domain: str | None,
    purpose: str | None,
    adversarial_threshold: float,
    embedding_model_name: str = DEFAULT_MODEL,
) -> tuple[list[dict], dict]:
    accepted_chunks: list[dict] = []
    accepted_documents: list[dict] = []
    rejected_documents: list[dict] = []

    for user_doc in user_docs:
        user_doc_domain = infer_user_doc_domain(
            domain=domain,
            purpose=f"{purpose or ''} {user_doc.name}",
        )
        chunks, validation = prepare_user_doc_chunks(
            user_doc=user_doc,
            domain=user_doc_domain,
            purpose=purpose,
            adversarial_threshold=adversarial_threshold,
            embedding_model_name=embedding_model_name,
            fail_on_flagged=False,
        )
        if validation["defense_passed"]:
            accepted_chunks.extend(chunks)
            accepted_documents.append(validation)
        else:
            rejected_documents.append(validation)

    validation_report = {
        "mode": "multi_user_doc_priority_retrieval",
        "domain": domain or "inferred_per_document",
        "user_doc_purpose": purpose,
        "total_documents": len(user_docs),
        "accepted_documents": len(accepted_documents),
        "rejected_documents": len(rejected_documents),
        "accepted_chunks": len(accepted_chunks),
        "adversarial_threshold": adversarial_threshold,
        "accepted": accepted_documents,
        "rejected": rejected_documents,
    }
    if not accepted_chunks:
        raise ValueError("All user documents were rejected; no user-upload context is available for generation.")
    return accepted_chunks, validation_report


def filter_chunks_by_domain(chunks: list[dict], domain: str | None) -> list[dict]:
    if not domain:
        return chunks
    return [chunk for chunk in chunks if chunk.get("domain") == domain]


def merge_clean_and_user_chunks(
    clean_chunks: list[dict],
    user_chunks: list[dict],
    top_k: int,
    max_user_chunks: int = 2,
) -> list[dict]:
    user_budget = min(max_user_chunks, len(user_chunks), max(top_k - 1, 0))
    clean_budget = max(top_k - user_budget, 0)
    selected: list[dict] = []
    used_ids: set[str] = set()

    for chunk in clean_chunks[:clean_budget] + user_chunks[:user_budget] + clean_chunks[clean_budget:]:
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id in used_ids:
            continue
        selected.append(chunk)
        used_ids.add(chunk_id)
        if len(selected) >= top_k:
            break

    return selected


def normalize_user_doc_path(path: Path) -> Path:
    if path.exists():
        return path
    normalized = Path(str(path).replace("\\", "/"))
    return normalized if normalized.exists() else path


def generate_with_groq(
    prompt: str,
    model: str = DEFAULT_GROQ_MODEL,
    max_output_tokens: int = 500,
    temperature: float = DEFAULT_GENERATION_TEMPERATURE,
) -> str:
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("Groq generation requires `groq`. Install with `pip install -r requirements.txt`.") from exc

    client = Groq()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=max_output_tokens,
    )
    return (completion.choices[0].message.content or "").strip()


def generate_with_openai(
    prompt: str,
    model: str = DEFAULT_OPENAI_MODEL,
    max_output_tokens: int = 500,
    temperature: float = DEFAULT_GENERATION_TEMPERATURE,
) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    return response.choices[0].message.content.strip()


def generate_with_local(
    prompt: str,
    model: str = DEFAULT_LOCAL_MODEL,
    endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    max_output_tokens: int = 500,
    temperature: float = DEFAULT_GENERATION_TEMPERATURE,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_output_tokens},
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
    temperature: float = DEFAULT_GENERATION_TEMPERATURE,
) -> str:
    if provider == "groq":
        return generate_with_groq(
            prompt,
            model=model or DEFAULT_GROQ_MODEL,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    if provider == "local":
        return generate_with_local(
            prompt,
            model=model or DEFAULT_LOCAL_MODEL,
            endpoint=local_endpoint,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    if provider == "openai":
        return generate_with_openai(
            prompt,
            model=model or DEFAULT_OPENAI_MODEL,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    raise ValueError("Unsupported generation provider. Expected 'groq', 'local', or 'openai'.")


def generate_response(
    question: str,
    domain: str | None = None,
    top_k: int = 8,
    persona: dict | None = None,
    provider: str = "groq",
    model: str | None = None,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    user_doc_chunks: list[dict] | None = None,
    user_doc_validation: dict | None = None,
    min_similarity_score: float | None = 0.18,
    temperature: float = DEFAULT_GENERATION_TEMPERATURE,
    dry_run: bool = False,
) -> dict:
    retrieval_domain = infer_domain_from_question(question, domain=domain)
    retrieval_query = f"Survey question: {question}"
    if user_doc_chunks:
        domain_user_chunks = filter_chunks_by_domain(user_doc_chunks, retrieval_domain)
        prioritized_user_chunks = score_chunks_lexical(question, domain_user_chunks)
        clean_chunks = retrieve_chunks(
            query=retrieval_query,
            top_k=top_k,
            domain=retrieval_domain,
            min_score=min_similarity_score,
        )
        chunks = merge_clean_and_user_chunks(
            clean_chunks=clean_chunks,
            user_chunks=prioritized_user_chunks,
            top_k=top_k,
        )
    else:
        chunks = retrieve_chunks(
            query=retrieval_query,
            top_k=top_k,
            domain=retrieval_domain,
            min_score=min_similarity_score,
        )
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
        temperature=temperature,
    )

    return {
        "persona_id": persona.get("persona_id") if persona else None,
        "question": question,
        "domain": domain,
        "retrieval_domain": retrieval_domain,
        "retrieval_query": retrieval_query,
        "persona": persona,
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "temperature": temperature,
        "retrieved_sources": source_refs(chunks),
        "retrieved_context": retrieved_context,
        "user_doc_validation": user_doc_validation,
        "prompt": prompt,
        "response": response_text,
        "dry_run": dry_run,
    }


def generate_responses(
    questions: list[str],
    personas: list[dict],
    domain: str | None = None,
    top_k: int = 8,
    provider: str = "groq",
    model: str | None = None,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    user_doc_chunks: list[dict] | None = None,
    user_doc_validation: dict | None = None,
    min_similarity_score: float | None = 0.18,
    temperature: float = DEFAULT_GENERATION_TEMPERATURE,
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
                    min_similarity_score=min_similarity_score,
                    temperature=temperature,
                    dry_run=dry_run,
                )
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic survey response using RAG context.")
    parser.add_argument("--question", action="append", help="Survey question to answer. Can be provided multiple times.")
    parser.add_argument("--questions-file", type=Path, help="Text file with one question per line, or JSON list.")
    parser.add_argument("--domain", choices=("ecommerce", "finance", "healthcare"), help="Optional RAG domain filter.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of RAG chunks to retrieve.")
    parser.add_argument(
        "--min-similarity-score",
        type=float,
        default=0.18,
        help="Minimum retrieval similarity score. Use 0 to disable filtering.",
    )
    parser.add_argument("--persona-json", help="Optional persona as a JSON object string.")
    parser.add_argument("--persona-file", type=Path, help="Optional JSONL persona file. Uses the first record.")
    parser.add_argument("--personas", type=Path, default=personas_path(), help="Personas JSON file.")
    parser.add_argument("--limit-personas", type=int, help="Only generate for the first N personas.")
    parser.add_argument("--provider", default="groq", choices=GENERATION_PROVIDERS, help="Generation provider.")
    parser.add_argument("--model", default=None, help="Generation model. Defaults depend on provider.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_GENERATION_TEMPERATURE,
        help="Generation temperature. Lower values usually improve grounding.",
    )
    parser.add_argument("--local-endpoint", default=DEFAULT_LOCAL_ENDPOINT, help="Local Ollama-compatible chat endpoint.")
    parser.add_argument(
        "--user-doc",
        type=Path,
        action="append",
        help="Optional user document (.pdf, .txt, .md, .docx, .json) to validate and prioritize. Can be repeated.",
    )
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
        args.user_doc = [normalize_user_doc_path(path) for path in args.user_doc]
        missing_docs = [path for path in args.user_doc if not path.exists()]
        if missing_docs:
            raise FileNotFoundError(f"User document does not exist: {missing_docs[0]}")
        user_doc_chunks, user_doc_validation = prepare_user_docs_chunks(
            user_docs=args.user_doc,
            domain=args.domain,
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
        min_similarity_score=args.min_similarity_score if args.min_similarity_score > 0 else None,
        temperature=args.temperature,
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
