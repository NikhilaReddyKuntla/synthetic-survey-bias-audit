# Project Instructions

This repository builds a synthetic survey pipeline with four main stages:

1. Generate demographic personas from ACS data.
2. Ingest and chunk trusted PDF source documents for Retrieval-Augmented Generation (RAG).
3. Embed the chunks into a FAISS vector index for semantic retrieval.
4. Retrieve context and generate survey responses for one or more personas.

The current implemented pipeline lives mainly in `src/persona`, `src/rag`, `src/generation`, and `src/utils`.

## Execution Flow

### End-to-end flow

1. `src/persona/generate_personas.py`
   Creates synthetic persona records from ACS-based demographic distributions.
   Output: `data/personas/personas.json`

2. `src/rag/ingest.py`
   Reads PDFs from `data/raw_sources/...`, extracts text, optionally summarizes charts/tables with a vision model, and writes chunked records.
   Output: `data/rag_docs/rag_chunks.jsonl`

3. `src/rag/embed.py`
   Converts chunk text into embeddings and stores them in a FAISS index plus metadata.
   Output: `vector_store/rag_index.faiss` and `vector_store/rag_metadata.json`

4. `src/rag/retrieve.py`
   Loads the FAISS index and returns the most relevant chunks for a query.
   Output: printed JSON retrieval results

5. `src/generation/generate_responses.py`
   Retrieves relevant context for each question, builds prompts, and calls Groq, OpenAI, or a local model.
   Output: `data/outputs/synthetic_responses.json`

### Dependency chain

- Persona generation depends on ACS CSVs in `data/raw_sources/acs/`.
- RAG ingestion depends on PDFs in `data/raw_sources/ecommerce/`, `finance/`, and `healthcare/`.
- Embedding depends on the chunk file created by ingestion.
- Retrieval depends on the FAISS index and metadata created by embedding.
- Response generation depends on personas plus the vector store.

## File-by-File Guide

### `src/persona/generate_personas.py`

Purpose:
Generates synthetic respondent personas using weighted sampling from ACS demographic distributions.

Functions:

- `acs_file(file_name)`
  Returns the full path to an ACS CSV under `data/raw_sources/acs/`.

- `load_demographic_distributions()`
  Reads ACS tables and creates sampling distributions for age, gender, race/ethnicity, income, education, and employment.

- `build_profile_text(persona)`
  Creates a readable sentence describing a persona. This is later inserted into prompts.

- `generate_personas(count, seed=None)`
  Samples `count` personas from the demographic distributions and assigns `persona_id` values.

- `save_personas(personas, output_path=None)`
  Writes the persona list to JSON.

- `parse_args()`
  Defines CLI flags such as `--n`, `--seed`, and `--output`.

- `main()`
  CLI entry point. Generates personas and saves them.

### `src/rag/ingest.py`

Purpose:
Builds the RAG corpus by reading PDFs, cleaning text, chunking content, and optionally adding visual summaries.

Functions:

- `infer_doc_type(file_name)`
  Heuristically tags a PDF as `data_book`, `health_stats`, `quarterly_report`, `report`, or generic `document`.

- `slugify(value)`
  Normalizes file names for stable chunk IDs.

- `infer_domain(pdf_path, domain=None)`
  Determines the domain for a PDF, either from `--domain` or the parent folder name.

- `discover_source_pdfs(input_paths=None, domain=None)`
  Finds PDFs to ingest. If no input is passed, it scans the default raw source folders.

- `build_pdf_chunks(...)`
  Core ingestion routine for one PDF:
  extracts page text, cleans it, chunks it, optionally summarizes pages with visuals, and returns structured chunk dictionaries.

- `build_chunks(...)`
  Runs `build_pdf_chunks()` for every discovered PDF.

- `merge_chunks(existing_chunks, new_chunks)`
  Upserts chunks by `chunk_id` when append mode is used.

- `ingest_documents(...)`
  Top-level orchestrator that builds chunks and writes JSONL output.

- `parse_args()`
  Defines CLI options for chunk sizing, inputs, append mode, vision provider, and output path.

- `main()`
  CLI entry point. Writes chunk JSONL.

Important notes:

- Allowed domains are `ecommerce`, `finance`, and `healthcare`.
- Visual summaries are optional and require either Groq, OpenAI, or a local vision model.

### `src/rag/embed.py`

Purpose:
Turns chunk text into embeddings and stores them in a FAISS vector index.

Functions:

- `load_embedding_model(model_name)`
  Loads a Sentence-Transformers model. It first tries a normal load, then retries with `local_files_only=True`.

- `embed_chunks(chunks_path=None, model_name=DEFAULT_MODEL, output_dir=None, append=False)`
  Reads chunk JSONL, encodes chunk text, builds or appends to a FAISS inner-product index, and writes metadata JSON.

- `parse_args()`
  Defines CLI flags like `--chunks`, `--model`, `--output-dir`, and `--append`.

- `main()`
  CLI entry point. Writes index and metadata.

Important notes:

- Default embedding model is `all-MiniLM-L6-v2`.
- First-time model download may require internet access in a normal runtime environment.

### `src/rag/retrieve.py`

Purpose:
Searches the FAISS store for the most relevant chunks to a query.

Functions:

- `load_vector_store(index_path=None, metadata_path=None)`
  Loads the FAISS index and its metadata records.

- `retrieve_chunks(query, top_k=3, domain=None, model_name=DEFAULT_MODEL, index_path=None, metadata_path=None)`
  Embeds the query, searches the index, optionally filters by domain, and returns the top chunk records.

- `format_retrieved_context(results)`
  Converts retrieved chunks into a prompt-ready text block including source metadata.

- `parse_args()`
  Defines CLI retrieval options.

- `main()`
  CLI entry point. Prints retrieval results as JSON.

### `src/rag/vision.py`

Purpose:
Adds optional visual understanding for PDF pages that contain charts, tables, or figures.

Functions:

- `_image_url_from_pdf_page(pdf_path, page_number, zoom=1.5)`
  Renders a PDF page to PNG bytes and returns a base64 data URL.

- `_image_base64_from_pdf_page(pdf_path, page_number, zoom=1.5)`
  Returns a raw base64 PNG string for local vision APIs.

- `summarize_visual_page_with_groq(...)`
  Sends a rendered page image to a Groq vision model and returns a concise summary.

- `summarize_visual_page_with_local(...)`
  Sends a rendered page image to a local Ollama-compatible vision endpoint.

- `summarize_visual_page_with_openai(...)`
  Sends a rendered page image to the OpenAI Responses API.

- `summarize_visual_page(...)`
  Provider router that chooses Groq, local, or OpenAI vision handling.

### `src/generation/generate_responses.py`

Purpose:
Builds RAG-backed prompts and generates survey answers for one or many personas.

Functions:

- `load_persona(persona_json=None, persona_file=None)`
  Loads a single persona from inline JSON or from the first row in a JSONL file.

- `load_personas(path)`
  Loads the main personas JSON file.

- `load_questions(question=None, questions_file=None)`
  Loads survey questions from repeated CLI flags, a text file, or a JSON list file.

- `source_refs(chunks)`
  Produces lightweight source metadata to store alongside each generated response.

- `generate_with_groq(prompt, model=DEFAULT_GROQ_MODEL)`
  Calls a Groq chat model.

- `generate_with_openai(prompt, model=DEFAULT_OPENAI_MODEL)`
  Calls the OpenAI Responses API.

- `generate_with_local(prompt, model=DEFAULT_LOCAL_MODEL, endpoint=DEFAULT_LOCAL_ENDPOINT)`
  Calls a local Ollama-compatible chat endpoint.

- `call_generation_model(prompt, provider="groq", model=None, local_endpoint=...)`
  Chooses the generation backend.

- `generate_response(...)`
  Retrieves relevant context, builds the prompt, optionally calls an LLM, and returns a full record for one persona/question pair.

- `generate_responses(...)`
  Loops over all questions and personas and collects response records.

- `parse_args()`
  Defines CLI options for questions, personas, retrieval settings, provider selection, output path, and dry-run mode.

- `main()`
  CLI entry point. Writes generated records to JSON.

Important notes:

- `--dry-run` is useful for testing prompt construction without spending API calls.
- The script assumes the vector store already exists.

## Utility Modules

### `src/utils/helpers.py`

Purpose:
Shared file path helpers and JSON/JSONL read-write helpers.

Functions:

- `project_root()`: Returns the repo root.
- `data_dir()`: Returns `data/`.
- `rag_docs_path()`: Returns the default chunk JSONL path.
- `personas_path()`: Returns the default personas JSON path.
- `vector_store_dir()`: Returns `vector_store/`.
- `outputs_dir()`: Returns `data/outputs/`.
- `synthetic_responses_path()`: Returns the default output JSON path.
- `ensure_parent_dir(path)`: Creates a file's parent directory if needed.
- `ensure_dir(path)`: Creates a directory if needed.
- `write_jsonl(path, rows)`: Writes rows as JSONL.
- `append_jsonl(path, rows)`: Appends rows as JSONL.
- `write_json(path, payload)`: Writes formatted JSON.
- `read_json(path)`: Reads JSON.
- `read_jsonl(path)`: Reads JSONL into a list of dicts.

### `src/utils/text_utils.py`

Purpose:
Text cleaning and chunking helpers for ingestion.

Functions:

- `normalize_whitespace(text)`
  Collapses repeated whitespace.

- `clean_text(raw_text)`
  Removes empty lines, repetitive headers/footers, and standalone page numbers.

- `chunk_text(text, chunk_size=700, chunk_overlap=120)`
  Splits text into overlapping word chunks.

- `extract_year(text)`
  Pulls a `20xx` year from text when present.

### `src/utils/pdf_utils.py`

Purpose:
PDF extraction and rendering utilities.

Functions:

- `extract_pdf_text(pdf_path)`
  Extracts joined text from all pages with text.

- `extract_pdf_pages_text(pdf_path)`
  Returns one text string per page.

- `page_has_visual_content(pdf_path, page_index, page_text="")`
  Detects whether a page likely contains charts/tables/images.

- `render_pdf_page_to_png_bytes(pdf_path, page_index, zoom=2.0)`
  Renders a PDF page to PNG bytes for vision models.

### `src/utils/prompt_templates.py`

Purpose:
Prompt formatting for survey response generation.

Functions:

- `format_persona(persona=None)`
  Converts persona data into prompt text.

- `build_survey_response_prompt(question, retrieved_context, persona=None)`
  Creates the final survey-answer prompt used by generation models.

### `src/utils/acs_utils.py`

Purpose:
Helpers for reading ACS CSVs and turning them into weighted sampling distributions.

Functions:

- `clean_label(value)`
  Normalizes ACS label strings.

- `parse_numeric(value)`
  Converts strings like percentages or comma-separated counts into floats.

- `read_acs_rows(path)`
  Reads ACS CSV rows.

- `row_label(row)`
  Extracts the grouping label from a row.

- `find_rows(path, labels)`
  Filters ACS rows down to specific labels.

- `value_from_first_matching_column(row, column_fragments)`
  Finds the first usable numeric value from matching columns.

- `distribution_from_labels(path, labels, column_fragments, fallback)`
  Builds a label-to-weight distribution.

- `sample_weighted(distribution, rng)`
  Randomly samples one label using the distribution weights.

## Task 3 Adversarial Pipeline

This repo also includes a Task 3 adversarial attack and defense pipeline. Its purpose is to test whether poisoned documents can manipulate generated survey responses, and whether the defense layer reduces that effect.

### Task 3 flow

1. `src/adversarial/generate_attacks.py`
   Generates poisoned attack documents for one or more domains.
   Output: `data/user_docs/attacks/attack_documents.json`

2. `src/adversarial/validate_docs.py`
   Scores attack documents against trusted RAG content, assigns trust levels, and flags suspicious chunks.
   Output: `data/outputs/adversarial_validation_report.json`

3. `src/adversarial/run_attack_experiment.py`
   Runs two conditions:
   clean RAG and poisoned RAG with defense.
   Outputs:
   `data/outputs/attack_clean.json`
   `data/outputs/attack_poisoned_with_defense.json`
   `data/outputs/attack_analysis.csv`
   `data/outputs/attack_analysis_report.md`

### `src/adversarial/generate_attacks.py`

Purpose:
Generates adversarial documents in the Task 3 schema and saves them under `data/user_docs/attacks/`.

Functions:

- `attack_docs_dir()`
  Returns `data/user_docs/attacks`.

- `default_attack_docs_path()`
  Returns the default output path for generated attack docs.

- `_normalize_domain(domain)`
  Normalizes domain strings into lowercase underscore format.

- `build_poisoned_text(domain, attack_type, target_claim)`
  Builds poisoned text for one attack strategy.

- `build_attack_documents(domains, target_claims=None)`
  Creates attack docs with `domain`, `attack_type`, `target_claim`, `poisoned_text`, and `source_file`.

- `infer_domains_from_clean_metadata(clean_metadata_path)`
  Infers available domains from trusted metadata when `--domain` is not supplied.

- `parse_args()`
  Defines CLI options for domain, clean metadata path, output path, and chunk preview output.

- `main()`
  CLI entry point. Writes attack documents and a RAG-compatible chunk preview.

### `src/adversarial/validate_docs.py`

Purpose:
Validates attack documents against trusted chunks and assigns trust levels before they are used in defended retrieval.

Functions:

- `_jaccard_similarity(tokens_a, tokens_b)`
  Computes token overlap for heuristic similarity.

- `_group_trusted_chunks_by_domain(trusted_chunks)`
  Organizes trusted chunks by domain for defense checks.

- `_contains_absolute_language(text)`
  Flags overgeneralized language such as sweeping or absolute claims.

- `_contains_unverified_statistical_claim(text)`
  Detects statistical-looking claims using heuristics.

- `evaluate_attack_document(attack_doc, trusted_chunks_by_domain)`
  Scores one attack document and returns trust details.

- `validate_attack_documents(attack_docs, trusted_chunks)`
  Builds the full trust report for all attack docs.

- `filter_attack_docs_by_trust(report, minimum_trust="medium")`
  Keeps only docs at or above the chosen trust threshold.

- `parse_args()`
  Defines CLI options for input docs, trusted chunks, minimum trust, and output report.

- `main()`
  CLI entry point. Writes `data/outputs/adversarial_validation_report.json`.

### `src/adversarial/run_attack_experiment.py`

Purpose:
Runs the end-to-end Task 3 experiment across clean and defended conditions.

Functions:

- `_write_markdown(path, content)`
  Writes the final markdown report.

- `_claim_mentioned_in_response(response_text, attack_docs)`
  Checks whether poisoned claims appear in a response.

- `_load_or_generate_attack_docs(args)`
  Loads existing attack docs or regenerates them.

- `_build_attack_analysis_report(rows, validation_report, semantic_threshold)`
  Builds the attack analysis markdown summary.

- `resolve_personas(args)`
  Reuses persona-loading logic for the experiment.

- `generate_response_with_store(...)`
  Generates one response against a specified vector store.

- `parse_args()`
  Defines CLI options for questions, personas, domain, provider, attack docs, output paths, and dry-run mode.

- `main()`
  Orchestrates the full experiment and writes JSON, CSV, and markdown outputs.

### `src/attacks/poison_utils.py`

Purpose:
Provides shared poisoning, retrieval fallback, scoring, and CSV helper utilities used by Task 3.

Functions:

- `infer_domains_from_metadata(metadata)`
  Extracts domain names from trusted metadata.

- `build_adversarial_templates(domain, target_product, competitor_product)`
  Legacy template builder for poisoned content.

- `_unique_chunk_id(base_chunk_id, used_chunk_ids)`
  Ensures poisoned chunk IDs stay unique.

- `build_poison_chunks(...)`
  Builds poisoned chunk dicts from templates.

- `attack_docs_to_poison_chunks(attack_docs, existing_chunk_ids=None)`
  Converts Task 3 attack docs into chunk records.

- `create_poisoned_vector_store(...)`
  Copies the clean store and appends poisoned content.

- `tokenize(text)`
  Tokenization helper for heuristic checks.

- `retrieve_chunks_lexical(query, metadata, top_k=3, domain=None)`
  Fast lexical retrieval fallback used in smoke tests.

- `hash_embed_texts(texts, dimension)`
  Produces offline hash embeddings for fast smoke runs.

- `keyword_overlap_score(baseline_text, attacked_text)`
  Measures lexical overlap between two responses.

- `semantic_shift_pct(response_similarity)`
  Converts similarity into semantic shift percentage.

- `response_similarity_score(...)`
  Computes cosine-style similarity between response pairs.

- `evaluate_response_shift(...)`
  Returns the main attack-shift metrics bundle.

- `_to_csv_value(value)`
  Serializes values for CSV writing.

- `write_rows_to_csv(path, rows, fieldnames=None)`
  Writes detailed attack-analysis rows to CSV.

### Task 3 execution summary

1. Generate attack docs.
2. Validate them with the defense gate.
3. Build the defended store after removing low-trust attack docs.
4. Generate responses for clean and defended conditions.
5. Compute response-shift and poisoned-claim metrics.
6. Save JSON, CSV, and markdown reports.

## Files Present But Not Implemented Yet

- `src/bias/detect.py`
- `src/bias/correct.py`

These bias files are currently empty placeholders. They do not contribute to the active execution flow yet.

## Recommended Run Order

Use the modules in this order:

1. `src.persona.generate_personas`
2. `src.rag.ingest`
3. `src.rag.embed`
4. `src.rag.retrieve` for spot-checking
5. `src.generation.generate_responses`

See `COMMANDS.md` for exact commands.
