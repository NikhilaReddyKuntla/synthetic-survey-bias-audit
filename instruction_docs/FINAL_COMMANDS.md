# Final Commands

These commands are sized for final project evidence while keeping runtime and API usage reasonable. Run from the repository root.

```bash
cd synthetic-survey-bias-audit
source .venv/bin/activate
```

If needed, create the environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set hosted model keys:

```bash
export GROQ_API_KEY="your_groq_key"
export OPENAI_API_KEY="your_openai_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

Recommended final run sizes:

- Personas: `100`
- RAG survey generation: first `25` personas across `questions.txt`
- Adversarial experiment: `40` question-persona cases
- User-doc demo generation: first `10` personas

With the current `questions.txt` containing 2 questions, this means:

- RAG generation: `2 * 25 = 50` generation calls
- Main adversarial experiment: `40` cases, about `80` generation calls plus validation judge calls
- User-doc generation: `2 * 10 = 20` generation calls

## 1. Personas

```bash
python -m src.persona.generate_personas \
  --n 100 \
  --seed 42 \
  --output data/personas/personas.json
```

Output:

```text
data/personas/personas.json
```

## 2. Trusted RAG Corpus

```bash
python -m src.rag.ingest
```

Output:

```text
data/rag_docs/rag_chunks.jsonl
```

## 3. Trusted Vector Store

```bash
python -m src.rag.embed
```

Outputs:

```text
vector_store/rag_index.faiss
vector_store/rag_metadata.json
```

## 4. RAG Validation

Run representative retrieval checks for all three domains. These do not call an LLM.

```bash
python -m src.rag.retrieve \
  --query "How are households feeling about credit card debt and monthly bills?" \
  --top-k 5 \
  --domain finance \
  --output data/outputs/rag_validation/finance_retrieval_results.json
```

```bash
python -m src.rag.retrieve \
  --query "What trends are visible in health insurance coverage and access?" \
  --top-k 5 \
  --domain healthcare \
  --output data/outputs/rag_validation/healthcare_retrieval_results.json
```

```bash
python -m src.rag.retrieve \
  --query "How has ecommerce spending and online shopping changed recently?" \
  --top-k 5 \
  --domain ecommerce \
  --output data/outputs/rag_validation/ecommerce_retrieval_results.json
```

Outputs:

```text
data/outputs/rag_validation/finance_retrieval_results.json
data/outputs/rag_validation/healthcare_retrieval_results.json
data/outputs/rag_validation/ecommerce_retrieval_results.json
```

## 5. RAG-Backed Survey Generation

Use 25 personas for a final but controlled generation sample.

```bash
export GROQ_API_KEY="YOUR_API_KEY"
python -m src.generation.generate_responses \
  --questions-file questions.txt \
  --personas data/personas/personas.json \
  --limit-personas 25 \
  --provider groq \
  --model llama-3.1-8b-instant \
  --output data/outputs/generation/synthetic_responses.json
```

Output:

```text
data/outputs/generation/synthetic_responses.json
```

## 6. Bias Detection

These commands generate model-specific synthetic datasets and compare them against ACS demographic distributions.

```bash
python -m src.bias.generate_gpt_survey
```

```bash
python -m src.bias.generate_deepseek_survey
```

```bash
python -m src.bias.analyze_bias
```

Outputs:

```text
data/outputs/bias_validation/gpt_synthetic_survey.csv
data/outputs/bias_validation/deepseek_synthetic_survey.csv
data/outputs/bias_validation/model_comparison_report.json
```

## 7. Generate Attack Documents

```bash
python -m src.adversarial.generate_attacks
```

Outputs:

```text
data/user_docs/attacks/attack_documents.json
data/user_docs/attacks/attack_chunks_preview.json
```

## 8. Validate Attack Documents

```bash
python -m src.adversarial.validate_docs \
  --provider groq \
  --model llama-3.1-8b-instant \
  --judge-timeout 20 \
  --judge-min-confidence 0.70 \
  --output data/outputs/attack/adversarial_validation_report.json
```

Output:

```text
data/outputs/attack/adversarial_validation_report.json
```

## 9. Main Adversarial Experiment

Use 40 question-persona cases for final results without triggering a full 200-case run. Increase `--max-cases` only if you want stronger statistics and can absorb the extra API usage.

```bash
python -m src.adversarial.run_attack_experiment \
  --questions-file questions.txt \
  --personas data/personas/personas.json \
  --max-cases 40 \
  --provider groq \
  --model llama-3.1-8b-instant \
  --judge-timeout 20 \
  --judge-min-confidence 0.70
```

Outputs:

```text
data/outputs/attack/attack_clean.json
data/outputs/attack/attack_poisoned_with_defense.json
data/outputs/attack/adversarial_validation_report.json
data/outputs/attack/attack_analysis.csv
data/outputs/attack/attack_analysis_report.md
```

## 10. User Document Validation

```bash
python -m src.adversarial.upload_validate \
  --input data/user_docs/uploads/example.txt \
  --domain finance \
  --purpose finance_context \
  --provider groq \
  --model llama-3.1-8b-instant \
  --judge-timeout 20 \
  --judge-min-confidence 0.70 \
  --output data/outputs/user_uploads/user_upload_validation_report.json
```

Output:

```text
data/outputs/user_uploads/user_upload_validation_report.json
```

## 11. Generation With User Document

Use 10 personas for the user-document demo so it stays inexpensive but shows repeated behavior.

```bash
python -m src.generation.generate_responses \
  --questions-file questions.txt \
  --personas data/personas/personas.json \
  --limit-personas 10 \
  --provider groq \
  --model llama-3.1-8b-instant \
  --user-doc data/user_docs/uploads/example.txt \
  --user-doc-purpose finance_context \
  --judge-timeout 20 \
  --judge-min-confidence 0.70 \
  --output data/outputs/generation/synthetic_responses_with_user_doc.json
```

Output:

```text
data/outputs/generation/synthetic_responses_with_user_doc.json
```

## 12. Cost Controls

To reduce runtime and API usage:

- Lower RAG generation with `--limit-personas 10`.
- Lower adversarial experiment with `--max-cases 20`.
- Keep RAG validation as-is because it does not call an LLM.

To strengthen final results:

- Increase RAG generation to `--limit-personas 50`.
- Increase adversarial experiment to `--max-cases 80`.
- A full run with 100 personas and 2 questions is about 200 cases, which means about 400 generation calls for the adversarial experiment.
