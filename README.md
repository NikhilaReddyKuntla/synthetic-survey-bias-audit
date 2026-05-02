# Synthetic Survey Bias Audit

This project audits a synthetic-survey workflow built with persona prompting and retrieval-augmented generation (RAG). The central question is:

Can an LLM simulate survey responses from demographic personas while staying grounded in trusted context, and can the pipeline detect bias or manipulation risks before those responses are used for decisions?

The repository contains three connected parts:

1. A RAG-backed survey generator for Product X scenarios in finance, ecommerce, and healthcare.
2. A validation notebook for retrieval quality, response groundedness, user-upload grounding, bias summaries, and adversarial-defense outputs.
3. Optional bias and adversarial-defense experiments for measuring demographic skew and poisoned-document robustness.

## What We Built

### RAG Survey System

The RAG system ingests trusted source documents from:

```text
data/raw_sources/
```

Each domain also includes a synthetic Product X context document:

```text
data/raw_sources/finance/synthetic_product_x_context.md
data/raw_sources/ecommerce/synthetic_product_x_context.md
data/raw_sources/healthcare/synthetic_product_x_context.md
```

These documents define the Product X scenario and include canonical grounding phrases, such as `monthly cash flow`, `transparent pricing`, and `out-of-pocket cost estimates`, so generated responses can stay close to the retrieved context.

The RAG implementation:

- chunks PDFs, text files, and markdown files
- embeds chunks with SentenceTransformers
- stores vectors in FAISS
- retrieves with hybrid semantic plus lexical reranking
- limits duplicate/source domination in retrieved context
- preserves retrieval scores for analysis
- supports optional user-upload documents
- filters user-upload chunks by inferred domain so irrelevant uploads do not reduce grounding

### Persona-Based Generation

Personas are generated into:

```text
data/personas/personas.json
```

Survey questions live in:

```text
data/questions/questions.txt
questions.txt
```

Generation is handled by:

```text
src/generation/generate_responses.py
```

The prompt asks the model to answer as a respondent, reuse exact grounding phrases when natural, avoid unsupported statistics, and keep answers short.

### Validation Notebook

The main notebook is:

```text
notebooks/project_results_analysis.ipynb
```

It checks:

- output availability
- RAG corpus health
- RAG retrieval quality by domain
- Product X context ranking
- generation groundedness
- user-doc groundedness
- unsupported precise-number usage
- bias report outputs
- adversarial validation outputs
- final pass/review gates

Native-heavy checks such as FAISS index inspection and SentenceTransformer semantic scoring are off by default in the notebook to avoid local Jupyter kernel crashes. You can enable them in the setup cell:

```python
ENABLE_NATIVE_INDEX_CHECK = True
ENABLE_SEMANTIC_GENERATION_CHECKS = True
```

## Repository Layout

```text
src/rag/                  RAG ingest, embedding, retrieval, vision summaries
src/generation/           RAG-backed survey response generation
src/persona/              persona generation
src/bias/                 baseline model bias generation and analysis
src/adversarial/          attack-doc generation, validation, defended experiments
src/attacks/              poisoned-vector-store utilities and attack helpers
src/utils/                shared helpers, prompts, text/PDF utilities

data/raw_sources/         trusted source documents and Product X context docs
data/rag_docs/            processed RAG chunks
vector_store/             FAISS index and aligned metadata
data/personas/            generated personas
data/questions/           survey questions
data/user_docs/           benign uploads and generated attack documents
data/outputs/             generated outputs and validation reports
notebooks/                final validation notebook
instruction_docs/         command reference
```

## Setup

Run from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set provider keys as needed:

```bash
export OPENAI_API_KEY="your_openai_key"
export GROQ_API_KEY="your_groq_key"
```

OpenAI is the recommended provider for the current commands because it avoids Groq daily token limits during repeated project runs.

## Execution Process

### 1. Generate Personas

```bash
python -m src.persona.generate_personas \
  --n 100 \
  --seed 42 \
  --output data/personas/personas.json
```

### 2. Build the Trusted RAG Corpus

```bash
python -m src.rag.ingest \
  --chunk-size 450 \
  --chunk-overlap 90
```

Output:

```text
data/rag_docs/rag_chunks.jsonl
```

### 3. Build the Vector Store

```bash
python -m src.rag.embed
```

Outputs:

```text
vector_store/rag_index.faiss
vector_store/rag_metadata.json
```

### 4. Run RAG Retrieval Validation

Finance:

```bash
python -m src.rag.retrieve \
  --query "For Product X Finance, how do monthly cash flow, bill reminders, savings realism, debt pressure, privacy, and account-linking security affect whether this customer would choose the product?" \
  --domain finance \
  --top-k 8 \
  --output data/outputs/rag_validation/finance_retrieval_results.json
```

Ecommerce:

```bash
python -m src.rag.retrieve \
  --query "For Product X Ecommerce, how do transparent pricing, delivery reliability, return policy clarity, review trust, hidden fees, and buyer protection affect whether this customer would choose the product?" \
  --domain ecommerce \
  --top-k 8 \
  --output data/outputs/rag_validation/ecommerce_retrieval_results.json
```

Healthcare:

```bash
python -m src.rag.retrieve \
  --query "For Product X Healthcare, how do insurance coverage, out-of-pocket cost estimates, provider availability, privacy protection, appointment preparation, and safety limitations affect whether this customer would choose the product?" \
  --domain healthcare \
  --top-k 8 \
  --output data/outputs/rag_validation/healthcare_retrieval_results.json
```

### 5. Generate RAG Survey Responses

Use a small run while iterating:

```bash
python -m src.generation.generate_responses \
  --questions-file data/questions/questions.txt \
  --personas data/personas/personas.json \
  --provider openai \
  --model gpt-4.1-mini \
  --top-k 6 \
  --temperature 0.25 \
  --limit-personas 2 \
  --output data/outputs/generation/synthetic_responses.json
```

Use a larger run for final reporting:

```bash
python -m src.generation.generate_responses \
  --questions-file data/questions/questions.txt \
  --personas data/personas/personas.json \
  --provider openai \
  --model gpt-4.1-mini \
  --top-k 8 \
  --temperature 0.35 \
  --limit-personas 25 \
  --output data/outputs/generation/synthetic_responses.json
```

### 6. Generate RAG Responses With User Documents

This tests whether added user documents improve context without lowering groundedness.

```bash
python -m src.generation.generate_responses \
  --questions-file data/questions/questions.txt \
  --personas data/personas/personas.json \
  --provider openai \
  --model gpt-4.1-mini \
  --top-k 6 \
  --temperature 0.25 \
  --limit-personas 2 \
  --user-doc data/user_docs/uploads/product_x_finance_budgeting.txt \
  --user-doc data/user_docs/uploads/product_x_finance_security.txt \
  --user-doc data/user_docs/uploads/product_x_ecommerce_trust.txt \
  --user-doc data/user_docs/uploads/product_x_healthcare_navigation.txt \
  --output data/outputs/generation/synthetic_responses_with_user_doc.json
```

The generator infers each upload's domain from the filename/purpose and only adds uploaded chunks that match the question domain.

### 7. Run the Validation Notebook

Open and run:

```text
notebooks/project_results_analysis.ipynb
```

Restart the kernel before running from the top if you previously hit a kernel crash.

## Optional Bias Analysis

The bias module generates baseline synthetic survey datasets and compares model outputs against benchmark demographic distributions.

```bash
python -m src.bias.generate_gpt_survey
python -m src.bias.generate_deepseek_survey
python -m src.bias.analyze_bias
```

Outputs:

```text
data/outputs/bias_validation/gpt_synthetic_survey.csv
data/outputs/bias_validation/deepseek_synthetic_survey.csv
data/outputs/bias_validation/model_comparison_report.json
```

## Optional Adversarial Defense Experiment

Generate attack documents:

```bash
python -m src.adversarial.generate_attacks \
  --high-count 2 \
  --medium-count 5 \
  --low-count 8
```

Validate attack documents:

```bash
python -m src.adversarial.validate_docs \
  --provider openai \
  --model gpt-4.1-mini \
  --judge-timeout 20 \
  --judge-min-confidence 0.70 \
  --output data/outputs/attack/adversarial_validation_report.json
```

Run the defended attack experiment:

```bash
python -m src.adversarial.run_attack_experiment \
  --questions-file data/questions/questions.txt \
  --personas data/personas/personas.json \
  --limit-personas 10 \
  --top-k 8 \
  --provider openai
```

Outputs:

```text
data/outputs/attack/adversarial_validation_report.json
data/outputs/attack/attack_clean.json
data/outputs/attack/attack_poisoned_with_defense.json
data/outputs/attack/attack_analysis.csv
data/outputs/attack/attack_analysis_report.md
```

## Key Metrics

The validation notebook reports:

- domain precision for retrieval
- Product X context hit rate and rank
- source diversity in retrieved context
- lexical/RAG groundedness
- ROUGE-L groundedness
- question relevance
- unsupported precise-number rate
- user-doc acceptance and grounding
- KL divergence for bias outputs
- adversarial document trust distribution
- poisoned-claim adoption in defended responses

## Common Issues

### Groq rate limits

If Groq returns a `429` token-per-day error, use OpenAI:

```bash
python -m src.generation.generate_responses \
  --questions-file data/questions/questions.txt \
  --provider openai \
  --model gpt-4.1-mini \
  --limit-personas 2 \
  --output data/outputs/generation/synthetic_responses.json
```

### OpenAI key missing

Set:

```bash
export OPENAI_API_KEY="your_openai_key"
```

### Shell line-continuation error

Backslashes must be the last character on the line:

```bash
python -m src.adversarial.run_attack_experiment \
  --questions-file data/questions/questions.txt \
  --personas data/personas/personas.json \
  --limit-personas 10 \
  --top-k 8 \
  --provider openai
```

### Notebook kernel crashes

Keep these defaults unless you need native checks:

```python
ENABLE_NATIVE_INDEX_CHECK = False
ENABLE_SEMANTIC_GENERATION_CHECKS = False
```

## Notes
- `notebooks/project_results_analysis.ipynb` is the main place to inspect project evidence.
- The fastest development loop is: update RAG docs or prompts, run ingest/embed, run retrieval validation, run a small OpenAI generation sample, then rerun the notebook.
