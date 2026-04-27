# Final Project Instructions

This project audits LLM-generated synthetic surveys for two risks:

1. Demographic bias in generated survey populations and answers.
2. Adversarial manipulation of RAG context through poisoned or user-uploaded documents.

The repository has five main workflows:

1. Bias detection.
2. Persona generation and survey response generation.
3. Trusted RAG document ingestion, embedding, and retrieval.
4. Adversarial testing against poisoned RAG context.
5. User document upload, validation, indexing, and priority retrieval.

Run commands from the repository root:

```bash
cd /Users/pgeesala/Desktop/synthetic-survey-bias-audit
```

Use module mode for source files:

```bash
python -m src.package.module
```

## 1. Project Agenda

The project simulates survey responses from synthetic personas, grounds responses in trusted documents through RAG, and then audits the outputs.

The main questions are:

- Do generated personas and responses reflect real demographic distributions?
- Do models overrepresent or underrepresent demographic groups?
- Does retrieved context improve grounded response generation?
- Can poisoned documents shift generated survey answers?
- Can a lightweight defense layer detect or reduce poisoned-context influence?
- Can user-uploaded documents be accepted only when they pass adversarial checks?

## 2. Repository Structure

Important source modules:

- `src/persona/generate_personas.py`: builds ACS-weighted synthetic personas.
- `src/rag/ingest.py`: extracts, cleans, chunks, and saves trusted PDF text.
- `src/rag/embed.py`: embeds chunks into a FAISS vector store.
- `src/rag/retrieve.py`: retrieves trusted context for survey questions.
- `src/generation/generate_responses.py`: builds persona + RAG prompts and calls an LLM.
- `src/bias/generate_gpt_survey.py`: generates a GPT synthetic survey dataset for bias comparison.
- `src/bias/generate_deepseek_survey.py`: generates a DeepSeek synthetic survey dataset for bias comparison.
- `src/bias/analyze_bias.py`: compares synthetic survey demographics against ACS benchmarks.
- `src/attacks/run_attack.py`: runs clean vs poisoned RAG response generation.
- `src/defense/defense.py`: filters poisoned retrieval sources and generates defended responses.
- `src/pipeline/document_pipeline.py`: handles user-upload document parsing, adversarial screening, indexing, and priority retrieval.

Important data locations:

- `data/raw_sources/acs/`: ACS demographic CSVs.
- `data/raw_sources/ecommerce/`: trusted ecommerce PDFs.
- `data/raw_sources/finance/`: trusted finance PDFs.
- `data/raw_sources/healthcare/`: trusted healthcare PDFs.
- `data/personas/personas.json`: generated persona file.
- `data/rag_docs/rag_chunks.jsonl`: trusted RAG chunks.
- `vector_store/rag_index.faiss`: trusted FAISS index.
- `vector_store/rag_metadata.json`: trusted chunk metadata.
- `data/outputs/generation/`: RAG-backed synthetic survey responses.
- `data/outputs/bias_validation/`: bias datasets and comparison report.
- `data/outputs/attack/`: adversarial validation, attack responses, and attack reports.
- `data/outputs/user_uploads/`: user-upload validation reports.
- `data/outputs/rag_validation/`: reserved for retrieval/RAG validation outputs.
- `results/`: legacy adversarial attack and defense outputs.

## 3. Environment

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Hosted model calls require API keys:

```bash
export OPENAI_API_KEY="your_openai_key"
export GROQ_API_KEY="your_groq_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

Local generation uses an Ollama-compatible endpoint:

```text
http://localhost:11434/api/chat
```

## 4. End-to-End Flow

The full project flow is:

1. Generate personas from ACS demographic distributions.
2. Ingest trusted PDFs into cleaned RAG chunks.
3. Embed chunks into a FAISS vector store.
4. Retrieve trusted context for survey questions.
5. Generate persona-grounded synthetic survey responses.
6. Generate model-specific survey datasets for bias analysis.
7. Compare synthetic distributions to ACS benchmarks.
8. Run poisoned-RAG adversarial testing.
9. Run defense filtering against poisoned retrieval.
10. Validate and optionally index user-uploaded documents.

## 5. Bias Detection

Bias detection compares synthetic survey records against ACS demographic distributions.

The implemented bias path uses:

- GPT synthetic survey output: `data/outputs/bias_validation/gpt_synthetic_survey.csv`
- DeepSeek synthetic survey output: `data/outputs/bias_validation/deepseek_synthetic_survey.csv`
- ACS benchmark CSVs in `data/raw_sources/acs/`

Then `src/bias/analyze_bias.py` computes:

- Race distribution gaps.
- Gender distribution gaps.
- Age distribution gaps.
- Income distribution gaps.
- Education distribution gaps.
- Employment distribution gaps.
- KL divergence.
- Chi-square goodness-of-fit p-values.
- Average response score by race.

Output:

- `data/outputs/bias_validation/model_comparison_report.json`

## 6. Persona + Generation Pipeline

The persona pipeline generates synthetic respondents using ACS-weighted sampling.

Each persona includes:

- `persona_id`
- `age_group`
- `gender`
- `race_ethnicity`
- `income_bracket`
- `education`
- `employment_status`
- `profile_text`

The response generation pipeline:

1. Loads questions.
2. Loads personas.
3. Retrieves relevant RAG chunks.
4. Builds a survey prompt containing persona + retrieved context.
5. Calls Groq, OpenAI, or local generation.
6. Writes structured output records.

Output:

- `data/outputs/generation/synthetic_responses.json`

The generation CLI also supports an optional `--user-doc` path. In that mode, the document is parsed, lightly screened for prompt-injection and known adversarial-template overlap, then its chunks are prioritized in memory before trusted RAG fallback context. This path does not write the user document into the main vector store.

## 7. Trusted RAG Pipeline

The trusted RAG pipeline uses curated PDFs from:

- `data/raw_sources/ecommerce/`
- `data/raw_sources/finance/`
- `data/raw_sources/healthcare/`

The ingestion step:

- discovers PDFs,
- infers domain from folder name,
- extracts page text,
- cleans repeated headers and page numbers,
- chunks text with overlap,
- optionally adds visual summaries,
- writes JSONL chunks.

The embedding step:

- loads `all-MiniLM-L6-v2` by default,
- encodes chunk text,
- normalizes embeddings,
- writes a FAISS inner-product index,
- writes matching metadata.

The retrieval step:

- embeds the query,
- searches FAISS,
- optionally filters by domain,
- returns top-k chunk records.

Primary outputs:

- `data/rag_docs/rag_chunks.jsonl`
- `vector_store/rag_index.faiss`
- `vector_store/rag_metadata.json`

## 8. Adversarial Testing

The implemented adversarial path uses:

- `src/attacks/run_attack.py`
- `src/defense/defense.py`

The attack script:

1. Starts from the clean trusted vector store.
2. Creates poisoned chunks using adversarial templates.
3. Appends poisoned chunks to a copied vector store.
4. Generates baseline responses using clean retrieval.
5. Generates attacked responses using poisoned retrieval.
6. Computes semantic and keyword shift.
7. Flags attack success when semantic shift is greater than the threshold.

Primary outputs:

- `results/poisoned_vector_store/rag_index.faiss`
- `results/poisoned_vector_store/rag_metadata.json`
- `results/poisoned_vector_store/injected_documents.json`
- `results/attack_responses.json`
- `results/attack_responses.csv`
- `results/attack_analysis.csv`

The defense script:

1. Loads attack response records.
2. Loads clean trusted chunk IDs.
3. Retrieves from the poisoned vector store.
4. Separates trusted and untrusted chunks.
5. Generates one response with poisoned context.
6. Generates one response with trusted-only filtered context.
7. Flags suspicious records when untrusted chunks appear or response divergence is high.
8. Uses filtered context for suspicious records.

Primary outputs:

- `results/defense_results.json`
- `results/defense_results.csv`

Important note:

Some older instruction docs mention newer modules such as `src.adversarial.generate_attacks`, `src.adversarial.run_attack_experiment`, and upload validation commands under `src.adversarial`. In this checkout, those files are not implemented, and `src/adversarial/validate_docs.py` is empty. Use the implemented `src.attacks.run_attack`, `src.defense.defense`, and `src.pipeline.document_pipeline` workflows unless those newer modules are restored.

## 9. User Document Upload Pipeline

The standalone user document pipeline is implemented in:

```text
src/pipeline/document_pipeline.py
```

It supports:

- `.pdf`
- `.txt`
- `.md`

The upload flow:

1. Parse the uploaded document.
2. Clean and chunk the text.
3. Build upload chunk metadata.
4. Compare uploaded chunks against known adversarial templates.
5. Reject the document if adversarial similarity exceeds the threshold.
6. Append accepted chunks to the selected vector store.
7. Retrieve uploaded chunks first for the current question.
8. Fall back to general trusted chunks.
9. Generate an answer from retrieved context.

The current standalone pipeline defense is similarity-based. The `generate_responses --user-doc` path uses a lighter lexical guard for prompt-injection phrases and adversarial-template overlap. These paths do not yet implement the richer trust gate described in the older docs, such as LLM judge scoring, purpose consistency, or medium/high trust levels.

## 10. Recommended Full Run Order

Use this order for a clean end-to-end run:

1. Set up environment.
2. Generate personas.
3. Ingest trusted RAG PDFs.
4. Build FAISS vector store.
5. Test retrieval.
6. Generate RAG-backed survey responses.
7. Generate GPT and DeepSeek synthetic survey datasets.
8. Run bias analysis.
9. Run adversarial attack simulation.
10. Run defense pass.
11. Run user document upload pipeline as needed.

## 11. Known Gaps

Current implementation gaps to remember:

- `src/adversarial/validate_docs.py` is empty.
- The old docs reference adversarial modules that are not currently present.
- The current attack pipeline is clean vs poisoned, while the newer desired evaluation is clean vs poisoned without defense vs poisoned with defense.
- User-upload validation is template-similarity based, not a full claim-support trust gate.
- `config.yaml` is currently empty.
