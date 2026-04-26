# Commands

Run all commands from the project root:

`/Users/pgeesala/Desktop/synthetic-survey-bias-audit`

Use module mode so imports like `from src...` work correctly:

`python -m ...`

## 1. Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional environment variables if using hosted models:

```bash
export OPENAI_API_KEY=your_key_here
export GROQ_API_KEY=your_key_here
```

## 2. Generate Personas

Generate the default personas file:

```bash
python -m src.persona.generate_personas
```

Generate a specific number with a fixed seed:

```bash
python -m src.persona.generate_personas --n 100 --seed 42
```

Write to a custom file:

```bash
python -m src.persona.generate_personas --n 250 --output data/personas/personas.json
```

## 3. Ingest RAG Documents

Ingest all PDFs from the default raw source folders:

```bash
python -m src.rag.ingest
```

Ingest a single folder explicitly:

```bash
python -m src.rag.ingest --input data/raw_sources/finance --domain finance
```

Ingest one PDF:

```bash
python -m src.rag.ingest --input "data/raw_sources/finance/SHED 2024.pdf" --domain finance
```

Append or upsert into an existing chunk file:

```bash
python -m src.rag.ingest --append
```

Enable visual summaries with Groq:

```bash
python -m src.rag.ingest --visual-summaries --vision-provider groq
```

Enable visual summaries with OpenAI:

```bash
python -m src.rag.ingest --visual-summaries --vision-provider openai
```

## 4. Build the Vector Store

Create embeddings and write the FAISS index:

```bash
python -m src.rag.embed
```

Use a custom chunk file:

```bash
python -m src.rag.embed --chunks data/rag_docs/rag_chunks.jsonl
```

Append only new chunks to an existing index:

```bash
python -m src.rag.embed --append
```

Use a different embedding model:

```bash
python -m src.rag.embed --model all-MiniLM-L6-v2
```

## 5. Test Retrieval

Spot-check retrieval results before generation:

```bash
python -m src.rag.retrieve --query "How are households feeling about credit card debt?" --top-k 3 --domain finance
```

Another example:

```bash
python -m src.rag.retrieve --query "What trends are visible in health insurance coverage?" --top-k 5 --domain healthcare
```

## 6. Generate Survey Responses

Generate responses for one question using the default personas file:

```bash
python -m src.generation.generate_responses --question "How confident are you in your household finances?" --domain finance --limit-personas 5 --provider groq
```

Generate from a questions text file:

```bash
python -m src.generation.generate_responses --questions-file questions.txt --domain healthcare --limit-personas 10 --provider groq
```

Generate using OpenAI:

```bash
python -m src.generation.generate_responses --question "How often do you shop online?" --domain ecommerce --limit-personas 5 --provider openai
```

Preview prompts without calling any model:

```bash
python -m src.generation.generate_responses --question "How secure do you feel financially?" --domain finance --limit-personas 2 --dry-run
```

Use a custom personas file:

```bash
python -m src.generation.generate_responses --question "How often do you use buy now, pay later?" --domain ecommerce --personas data/personas/personas.json --limit-personas 10
```

Generate responses while validating and prioritizing a user-uploaded document first:

```bash
python -m src.generation.generate_responses --question "How should we evaluate whether our survey system is grounded and robust?" --domain finance --limit-personas 5 --provider groq --user-doc /path/to/uploaded_document.pdf --user-doc-purpose "project plan"
```

## 7. Task 3 Adversarial Pipeline

Generate adversarial attack documents:

```bash
python -m src.adversarial.generate_attacks
```

Generate attack docs for one domain only:

```bash
python -m src.adversarial.generate_attacks --domain finance
```

Expected outputs:

- `data/user_docs/attacks/attack_documents.json`
- `data/user_docs/attacks/attack_chunks_preview.json`

Validate attack documents with the defense gate:

```bash
python -m src.adversarial.validate_docs
```

Expected output:

- `data/outputs/adversarial_validation_report.json`

Run the full clean vs poisoned vs defended experiment:

```bash
python -m src.adversarial.run_attack_experiment --questions-file data/questions/questions.txt --personas data/personas/personas.json --top-k 3 --provider groq
```

Fast smoke test without model calls:

```bash
python -m src.adversarial.run_attack_experiment --questions-file data/questions/questions.txt --personas data/personas/personas.json --top-k 3 --domain finance --fast-poison-vectors --dry-run
```

Force regeneration of attack docs during the run:

```bash
python -m src.adversarial.run_attack_experiment --questions-file data/questions/questions.txt --personas data/personas/personas.json --regenerate-attack-docs --provider groq
```

Expected Task 3 outputs:

- `data/outputs/attack_clean.json`
- `data/outputs/attack_poisoned_no_defense.json`
- `data/outputs/attack_poisoned_with_defense.json`
- `data/outputs/adversarial_validation_report.json`
- `data/outputs/attack_analysis.csv`
- `data/outputs/attack_analysis_report.md`

## Full Recommended Sequence

Run these one after the other:

```bash
source .venv/bin/activate
```

```bash
python -m src.persona.generate_personas --n 100 --seed 42
```

```bash
python -m src.rag.ingest
```

```bash
python -m src.rag.embed
```

```bash
python -m src.rag.retrieve --query "What are the main trends in this domain?" --top-k 3 --domain finance
```

```bash
python -m src.generation.generate_responses --question "How confident are you in your financial future?" --domain finance --limit-personas 5 --provider groq
```

## Output Files You Should Expect

- Personas: `data/personas/personas.json`
- RAG chunks: `data/rag_docs/rag_chunks.jsonl`
- Vector index: `vector_store/rag_index.faiss`
- Vector metadata: `vector_store/rag_metadata.json`
- Generated responses: `data/outputs/synthetic_responses.json`
- User validated chunks: `data/user_docs/validated_chunks.jsonl`
- User upload vector store: `vector_store/user_uploads/`
- Attack documents: `data/user_docs/attacks/attack_documents.json`
- Attack analysis report: `data/outputs/attack_analysis_report.md`

## Troubleshooting

If `python src/...` fails with import errors:

Use:

```bash
python -m src.persona.generate_personas
```

not:

```bash
python src/persona/generate_personas.py
```

If embedding model download fails on first run:

- Make sure the environment has internet access once, or that the model is already cached locally.

If vision summaries fail:

- Make sure `PyMuPDF` is installed.
- Make sure the selected provider API key is available.

If Task 3 commands fail:

- Make sure `vector_store/rag_index.faiss` and `vector_store/rag_metadata.json` already exist.
- Make sure `data/personas/personas.json` exists.
- Make sure `data/questions/questions.txt` exists if you use `--questions-file`.
