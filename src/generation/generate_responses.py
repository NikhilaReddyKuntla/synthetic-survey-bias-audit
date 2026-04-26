from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from sentence_transformers import SentenceTransformer

from src.rag.embed import DEFAULT_MODEL, load_embedding_model
from src.rag.retrieve import format_retrieved_context, retrieve_chunks, retrieve_chunks_with_priority
from src.utils.helpers import personas_path, read_json, read_jsonl, synthetic_responses_path, write_json
from src.utils.prompt_templates import build_survey_response_prompt

DEFAULT_LOCAL_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_LOCAL_MODEL = "llama3.1"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_JUDGE_TIMEOUT = 20
DEFAULT_JUDGE_MIN_CONFIDENCE = 0.70
GENERATION_PROVIDERS = ("groq", "deepseek", "local", "openai")


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
            "source_kind": chunk.get("source_kind"),
            "trust_score": chunk.get("trust_score"),
            "upload_id": chunk.get("upload_id"),
            "upload_purpose": chunk.get("upload_purpose"),
            "year": chunk.get("year"),
            "page_number": chunk.get("page_number"),
        }
        for chunk in chunks
    ]


def generate_with_groq(
    prompt: str,
    model: str = DEFAULT_GROQ_MODEL,
    max_output_tokens: int = 500,
) -> str:
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


def generate_with_openai(
    prompt: str,
    model: str = DEFAULT_OPENAI_MODEL,
    max_output_tokens: int = 500,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI generation requires `openai`. Install with `pip install -r requirements.txt`.") from exc

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.7,
        max_output_tokens=max_output_tokens,
    )
    return getattr(response, "output_text", "").strip()


def generate_with_deepseek(
    prompt: str,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    max_output_tokens: int = 500,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("DeepSeek generation requires `openai`. Install with `pip install -r requirements.txt`.") from exc

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DeepSeek generation requires DEEPSEEK_API_KEY to be set in your environment.")

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_output_tokens,
    )
    return (completion.choices[0].message.content or "").strip()


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
        return generate_with_groq(
            prompt,
            model=model or DEFAULT_GROQ_MODEL,
            max_output_tokens=max_output_tokens,
        )
    if provider == "deepseek":
        return generate_with_deepseek(
            prompt,
            model=model or DEFAULT_DEEPSEEK_MODEL,
            max_output_tokens=max_output_tokens,
        )
    if provider == "local":
        return generate_with_local(
            prompt,
            model=model or DEFAULT_LOCAL_MODEL,
            endpoint=local_endpoint,
            max_output_tokens=max_output_tokens,
        )
    if provider == "openai":
        return generate_with_openai(
            prompt,
            model=model or DEFAULT_OPENAI_MODEL,
            max_output_tokens=max_output_tokens,
        )
    raise ValueError(f"Unsupported generation provider. Expected one of: {', '.join(GENERATION_PROVIDERS)}.")


def generate_response(
    question: str,
    domain: str | None = None,
    top_k: int = 3,
    persona: dict | None = None,
    provider: str = "groq",
    model: str | None = None,
    embedding_model_name: str = DEFAULT_MODEL,
    embedding_model: SentenceTransformer | None = None,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    dry_run: bool = False,
    upload_purpose: str | None = None,
    upload_ids: list[str] | None = None,
) -> dict:
    if upload_purpose or upload_ids:
        chunks = retrieve_chunks_with_priority(
            query=question,
            top_k=top_k,
            domain=domain,
            model_name=embedding_model_name,
            embedding_model=embedding_model,
            upload_purpose=upload_purpose,
            upload_ids=upload_ids,
        )
    else:
        chunks = retrieve_chunks(
            query=question,
            top_k=top_k,
            domain=domain,
            model_name=embedding_model_name,
            embedding_model=embedding_model,
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
    )

    return {
        "persona_id": persona.get("persona_id") if persona else None,
        "question": question,
        "domain": domain,
        "persona": persona,
        "provider": provider,
        "model": model,
        "embedding_model": embedding_model_name,
        "top_k": top_k,
        "retrieved_sources": source_refs(chunks),
        "prompt": prompt,
        "response": response_text,
        "dry_run": dry_run,
        "upload_purpose": upload_purpose,
        "upload_ids": upload_ids or [],
    }


def generate_responses(
    questions: list[str],
    personas: list[dict],
    domain: str | None = None,
    top_k: int = 3,
    provider: str = "groq",
    model: str | None = None,
    embedding_model_name: str = DEFAULT_MODEL,
    local_endpoint: str = DEFAULT_LOCAL_ENDPOINT,
    dry_run: bool = False,
    upload_purpose: str | None = None,
    upload_ids: list[str] | None = None,
) -> list[dict]:
    records: list[dict] = []
    embedding_model = load_embedding_model(embedding_model_name)
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
                    embedding_model_name=embedding_model_name,
                    embedding_model=embedding_model,
                    local_endpoint=local_endpoint,
                    dry_run=dry_run,
                    upload_purpose=upload_purpose,
                    upload_ids=upload_ids,
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
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL, help="Embedding model for retrieval.")
    parser.add_argument("--local-endpoint", default=DEFAULT_LOCAL_ENDPOINT, help="Local Ollama-compatible chat endpoint.")
    parser.add_argument("--output", type=Path, default=synthetic_responses_path(), help="JSON output path.")
    parser.add_argument(
        "--user-doc",
        type=Path,
        action="append",
        dest="user_docs",
        help="User-uploaded document to validate and prioritize during retrieval. Can be provided multiple times.",
    )
    parser.add_argument(
        "--user-doc-purpose",
        default=None,
        help="Purpose label used to prioritize the current uploaded documents during retrieval.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional dedicated model for upload-defense adjudication.",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=DEFAULT_JUDGE_TIMEOUT,
        help="Upload-defense judge timeout in seconds.",
    )
    parser.add_argument(
        "--judge-min-confidence",
        type=float,
        default=DEFAULT_JUDGE_MIN_CONFIDENCE,
        help="Minimum judge confidence required for uploaded documents to be indexed.",
    )
    parser.add_argument(
        "--unsafe-allow-main-store-write",
        action="store_true",
        help="Allow upload-defense indexing to write to clean main store paths (unsafe, disabled by default).",
    )
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

    upload_ids: list[str] = []
    validation_report: dict | None = None
    if args.user_docs:
        try:
            from src.adversarial.upload_validate import validate_and_index_documents
        except ImportError as exc:
            raise RuntimeError(
                "User-upload validation is not available in the current environment. "
                "The base generation and Task 3 adversarial pipeline still work, but the "
                "`--user-doc` path requires `src.adversarial.upload_validate`."
            ) from exc

        validation_report = validate_and_index_documents(
            input_paths=args.user_docs,
            domain=args.domain,
            purpose=args.user_doc_purpose,
            provider=args.provider,
            model=args.model,
            judge_model=args.judge_model,
            local_endpoint=args.local_endpoint,
            judge_timeout=args.judge_timeout,
            judge_min_confidence=args.judge_min_confidence,
            unsafe_allow_main_store_write=args.unsafe_allow_main_store_write,
        )
        accepted_documents = validation_report.get("accepted_documents", [])
        upload_ids = [str(record.get("upload_id")) for record in accepted_documents if record.get("upload_id")]
        if not upload_ids:
            raise ValueError("All uploaded documents were rejected by the defense pipeline; nothing was indexed.")

    records = generate_responses(
        questions=questions,
        personas=personas,
        domain=args.domain,
        top_k=args.top_k,
        provider=args.provider,
        model=args.model,
        embedding_model_name=args.embedding_model,
        local_endpoint=args.local_endpoint,
        dry_run=args.dry_run,
        upload_purpose=args.user_doc_purpose,
        upload_ids=upload_ids,
    )
    if validation_report is not None:
        for record in records:
            record["upload_validation"] = {
                "accepted_upload_ids": upload_ids,
                "summary": validation_report.get("summary", {}),
                "report_path": str(validation_report.get("report_path")),
            }
    write_json(args.output, records)

    if args.dry_run:
        for record in records[:3]:
            print(record["prompt"])
            print("\n---\n")
    else:
        print(f"Generated {len(records)} synthetic responses.")
    if validation_report is not None:
        print(f"Validated {len(args.user_docs)} uploaded document(s).")
        print(f"Defense report: {validation_report['report_path']}")
    print(f"Wrote generation records to {args.output}")


if __name__ == "__main__":
    main()
