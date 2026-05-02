# Final Project Instructions

This project audits synthetic survey generation across three connected risks:

1. Bias in synthetic personas and survey answers.
2. Weak grounding in RAG-generated answers.
3. Adversarial or malicious documents entering the retrieval pipeline.

Run everything from the repository root:

```bash
cd /Users/pgeesala/Desktop/synthetic-survey-bias-audit
source .venv/bin/activate
```

Use module execution for source files:

```bash
python -m src.package.module
```

## Problem Statement

LLMs can generate useful synthetic survey responses, but their outputs may:

- overrepresent or underrepresent demographic groups,
- answer from model prior knowledge instead of project evidence,
- become vulnerable when retrieved context contains poisoned documents,
- trust user-uploaded files that include prompt injection or fake claims.

The project solves this by combining ACS-based persona generation, RAG grounding, model comparison for bias, adversarial document validation, and user-upload trust gating.

## Tools Used

- Python: main implementation language for reproducible CLI pipelines.
- Pandas and NumPy: demographic analysis, CSV processing, summary metrics.
- SciPy: chi-square and KL-divergence style bias measurements.
- SentenceTransformers: local embedding model for retrieval.
- FAISS: fast vector search over trusted and uploaded document chunks.
- OpenAI, Groq, DeepSeek: hosted LLM providers for response generation, baseline comparison, and defense judging.
- Matplotlib: final notebook visualizations.
- PyPDF/text parsing utilities: extracting source text from PDFs, TXT, and Markdown files.

Why these tools:

- FAISS plus SentenceTransformers keeps retrieval local, fast, and cheap.
- Hosted LLMs allow direct comparison between model families.
- ACS data gives a real demographic benchmark instead of evaluating bias only by intuition.
- A separate defense layer makes adversarial behavior measurable instead of hidden inside generation output.

## Main Data Sources

- `data/raw_sources/acs/`: ACS demographic benchmark CSVs.
- `data/raw_sources/ecommerce/`: ecommerce trusted source docs and synthetic Product X context.
- `data/raw_sources/finance/`: finance trusted source docs and synthetic Product X context.
- `data/raw_sources/healthcare/`: healthcare trusted source docs and synthetic Product X context.
- `data/user_docs/uploads/`: example user-upload documents, including benign and malicious files.
- `data/user_docs/attacks/`: generated adversarial test documents.

## Main Outputs

- `data/personas/personas.json`: ACS-weighted synthetic personas.
- `data/rag_docs/rag_chunks.jsonl`: trusted RAG chunks.
- `vector_store/rag_index.faiss`: trusted vector index.
- `vector_store/rag_metadata.json`: metadata aligned with the trusted FAISS index.
- `data/outputs/rag_validation/`: retrieval validation outputs.
- `data/outputs/generation/`: RAG-backed synthetic survey responses.
- `data/outputs/bias_validation/`: GPT/DeepSeek survey outputs and bias comparison report.
- `data/outputs/attack/`: adversarial validation and clean-vs-defended attack results.
- `data/outputs/user_uploads/`: user document validation reports.

## Source File Guide

### Persona Files

- `src/persona/generate_personas.py`
  - Reads ACS distributions.
  - Samples realistic demographic personas.
  - Writes `data/personas/personas.json`.
  - Used because final survey generation should start from controlled demographic profiles.

### RAG Files

- `src/rag/ingest.py`
  - Reads PDFs, TXT, and Markdown files from trusted domain folders.
  - Cleans and chunks text.
  - Tags chunks with domain, source file, document type, year, and chunk id.
  - Writes `data/rag_docs/rag_chunks.jsonl`.

- `src/rag/embed.py`
  - Loads `all-MiniLM-L6-v2` by default.
  - Embeds all RAG chunks.
  - Stores vectors in FAISS and metadata in JSON.
  - Used because retrieval needs fast semantic matching.

- `src/rag/retrieve.py`
  - Embeds a query.
  - Searches FAISS.
  - Supports `--domain` filtering and `--top-k`.
  - Writes retrieval validation files for notebook analysis.

- `src/rag/vision.py`
  - Optional visual-page summarization support for PDFs.
  - Useful only when PDF pages contain meaningful visual information.

### Generation Files

- `src/generation/generate_responses.py`
  - Loads questions and personas.
  - Retrieves top-k RAG chunks.
  - Builds a short survey-style prompt.
  - Calls OpenAI, Groq, or a local endpoint.
  - Supports repeated `--user-doc` uploads.
  - Validates uploaded docs before using them as priority context.
  - Writes synthetic responses to `data/outputs/generation/`.

- `src/utils/prompt_templates.py`
  - Central prompt builder.
  - Instructs the model to use provided context where relevant and keep answers short.
  - Used so generation, attack, and defended runs share consistent prompt behavior.

### Bias Files

- `src/bias/generate_gpt_survey.py`
  - Generates GPT baseline synthetic survey data.
  - Writes `gpt_synthetic_survey.csv`.

- `src/bias/generate_deepseek_survey.py`
  - Generates DeepSeek baseline synthetic survey data.
  - Writes `deepseek_synthetic_survey.csv`.

- `src/bias/analyze_bias.py`
  - Compares GPT and DeepSeek outputs against ACS distributions.
  - Measures demographic gaps, chi-square behavior, KL divergence, and response differences.
  - Writes `model_comparison_report.json`.

### Adversarial Files

- `src/adversarial/generate_attacks.py`
  - Creates synthetic attack documents.
  - Current final mix is `high=2`, `medium=5`, `low=8`.
  - High/medium docs are closer to trusted evidence; low docs contain unsupported, absolute, fake-stat, or injection-style content.

- `src/adversarial/defense_decision.py`
  - Core trust gate.
  - Checks prompt injection, unsupported target claims, absolute language, fake stats, and alignment with trusted chunks.
  - Uses an LLM judge when enabled.
  - Produces `low`, `medium`, or `high` trust plus reasons.

- `src/adversarial/validate_docs.py`
  - Runs the defense gate on generated attack documents.
  - Writes `data/outputs/attack/adversarial_validation_report.json`.
  - Preserves intended trust labels so the notebook can compare intended vs detected trust.

- `src/adversarial/run_attack_experiment.py`
  - Runs clean retrieval vs defended retrieval.
  - Builds poisoned vector stores only in controlled output locations.
  - Reports response shift separately from poisoned-claim appearance.
  - This matters because a shifted answer is not automatically a successful poisoned-claim adoption.

- `src/adversarial/upload_validate.py`
  - Validates one or more uploaded files.
  - Accepts trusted files, rejects malicious files, chunks accepted text, and indexes accepted chunks into the user-upload vector store.
  - Writes `data/outputs/user_uploads/user_upload_validation_report.json`.

### Attack Utility Files

- `src/attacks/poison_utils.py`
  - Shared helpers for poisoned vector-store creation, lexical retrieval, tokenization, semantic shift, and CSV writing.

- `src/attacks/run_attack.py`
  - Older attack runner retained for compatibility.
  - Prefer `src.adversarial.run_attack_experiment` for final results because it includes the newer trust-gated defense flow.

### Utility Files

- `src/utils/acs_utils.py`
  - Reads ACS CSV labels and samples weighted categories.

- `src/utils/doc_utils.py`
  - Extracts clean text from supported uploaded/source documents.

- `src/utils/pdf_utils.py`
  - Extracts text from PDF pages and detects visual-heavy pages.

- `src/utils/text_utils.py`
  - Cleans text, chunks text, and extracts years from filenames.

- `src/utils/helpers.py`
  - Central path helpers and JSON/JSONL read-write helpers.
  - Keeps output folders consistent.

## What Each Pipeline Proves

### Bias Detection

Shows whether generated synthetic survey datasets match ACS demographic benchmarks.

Final evidence:

- GPT baseline
- DeepSeek baseline
- ACS comparison metrics
- demographic distribution visualizations in the notebook

### RAG Grounding

Shows whether retrieved evidence is relevant to Product X questions across finance, healthcare, and ecommerce.

Final evidence:

- retrieval files under `data/outputs/rag_validation/`
- generated RAG survey responses under `data/outputs/generation/`
- notebook metrics for groundedness, relevance, source coverage, and response length

### Persona + Generation

Shows that responses are produced for demographically structured personas, not generic anonymous users.

Final evidence:

- `data/personas/personas.json`
- `data/outputs/generation/synthetic_responses.json`

### Adversarial Testing

Shows whether poisoned documents can influence generated answers and whether the defense prevents poisoned claims from appearing.

Important interpretation:

- `response_shift_rate` means the defended answer changed compared with the clean answer.
- `poisoned_claim_rate` means the attack claim actually appeared in the defended answer.
- A high response shift rate with a low poisoned claim rate means the defense changed behavior but blocked the malicious claim.

### User Document Upload

Shows that benign uploaded docs can be accepted while malicious uploaded docs are rejected.

Final evidence:

- 6 upload files tested
- 4 benign accepted
- 2 malicious rejected
- accepted chunks indexed only after validation

## Final `src` Execution Flow

Use this order when running source files for final results:

```text
src.persona.generate_personas
        |
        v
src.rag.ingest
        |
        v
src.rag.embed
        |
        v
src.rag.retrieve
        |
        v
src.generation.generate_responses
        |
        v
src.bias.generate_gpt_survey
        |
        v
src.bias.generate_deepseek_survey
        |
        v
src.bias.analyze_bias
        |
        v
src.adversarial.generate_attacks
        |
        v
src.adversarial.validate_docs
        |
        v
src.adversarial.run_attack_experiment
        |
        v
src.adversarial.upload_validate
        |
        v
notebooks/project_results_analysis.ipynb
```

Short version:

```text
ACS -> Personas -> Trusted Docs -> Chunks -> FAISS -> Retrieval -> RAG Responses
ACS + Model Outputs -> Bias Analysis
Attack Docs -> Defense Validation -> Attack Experiment
User Docs -> Upload Validation -> Accepted User Context
All Outputs -> Notebook Analysis
```

Use `instruction_docs/FINAL_COMMANDS.md` for exact runnable commands and recommended final sample sizes.
