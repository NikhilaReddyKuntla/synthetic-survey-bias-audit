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

With the current `questions.txt` containing 4 questions, this means:

- RAG generation: `4 * 25 = 100` generation calls
- Main adversarial experiment: `40` cases, about `80` generation calls plus validation judge calls
- User-doc generation: `4 * 10 = 40` generation calls

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

This ingests both real source PDFs and the synthetic Product X project-context documents:

```text
data/raw_sources/ecommerce/synthetic_product_x_context.md
data/raw_sources/finance/synthetic_product_x_context.md
data/raw_sources/healthcare/synthetic_product_x_context.md
```

Use shorter chunks so Product X context is more focused during retrieval.

```bash
python -m src.rag.ingest \
  --chunk-size 450 \
  --chunk-overlap 90
```

Output:

```text
data/rag_docs/rag_chunks.jsonl
```

Optional quick check:

```bash
grep -n "synthetic_product_x_context" data/rag_docs/rag_chunks.jsonl
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

Run Product X retrieval checks for all three domains. These do not call an LLM.

```bash
python -m src.rag.retrieve \
  --query "For Product X Finance, how do monthly cash flow, bill reminders, savings realism, debt pressure, privacy, and account-linking security affect whether this customer would choose the product?" \
  --top-k 8 \
  --min-score 0.18 \
  --domain finance \
  --output data/outputs/rag_validation/finance_retrieval_results.json
```

```bash
python -m src.rag.retrieve \
  --query "For Product X Healthcare, how do insurance coverage, out-of-pocket cost estimates, provider availability, privacy protection, appointment preparation, and safety limitations affect whether this customer would choose the product?" \
  --top-k 8 \
  --min-score 0.18 \
  --domain healthcare \
  --output data/outputs/rag_validation/healthcare_retrieval_results.json
```

```bash
python -m src.rag.retrieve \
  --query "For Product X Ecommerce, how do transparent pricing, delivery reliability, return policy clarity, review trust, hidden fees, and buyer protection affect whether this customer would choose the product?" \
  --top-k 8 \
  --min-score 0.18 \
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

Use 25 personas for a final but controlled generation sample. This is the main "Our RAG model" output for the notebook.

```bash
python -m src.generation.generate_responses \
  --questions-file questions.txt \
  --personas data/personas/personas.json \
  --limit-personas 25 \
  --top-k 8 \
  --min-similarity-score 0.18 \
  --provider openai \
  --model gpt-4.1-mini \
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
python -m src.adversarial.generate_attacks \
  --high-count 2 \
  --medium-count 5 \
  --low-count 8
```

Outputs:

```text
data/user_docs/attacks/attack_documents.json
data/user_docs/attacks/attack_chunks_preview.json
```

This creates 15 validation test documents with intended trust labels:

```text
high=2, medium=5, low=8
```

## 8. Validate Attack Documents

```bash
python -m src.adversarial.validate_docs \
  --provider openai \
  --model gpt-4.1-mini \
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
  --top-k 8 \
  --provider openai \
  --model gpt-4.1-mini \
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

Validate six uploaded documents: four benign Product X context notes and two malicious documents.

```bash
python -m src.adversarial.upload_validate \
  --input data/user_docs/uploads/product_x_finance_budgeting.txt \
  --input data/user_docs/uploads/product_x_finance_security.txt \
  --input data/user_docs/uploads/product_x_ecommerce_trust.txt \
  --input data/user_docs/uploads/product_x_healthcare_navigation.txt \
  --input data/user_docs/uploads/malicious_prompt_injection.txt \
  --input data/user_docs/uploads/malicious_fake_stats.txt \
  --domain finance \
  --purpose product_x_context_test \
  --provider openai \
  --model gpt-4.1-mini \
  --judge-timeout 20 \
  --judge-min-confidence 0.70 \
  --output data/outputs/user_uploads/user_upload_validation_report.json
```

Output:

```text
data/outputs/user_uploads/user_upload_validation_report.json
```

## 11. Generation With User Document

Use the same six uploaded documents for generation. The generation pipeline validates each upload first, prioritizes accepted upload chunks, and records accepted/rejected document results in `user_doc_validation`. The two malicious documents should be rejected and not used as context.

```bash
python -m src.generation.generate_responses \
  --questions-file questions.txt \
  --personas data/personas/personas.json \
  --limit-personas 10 \
  --top-k 8 \
  --min-similarity-score 0.18 \
  --user-doc data/user_docs/uploads/product_x_finance_budgeting.txt \
  --user-doc data/user_docs/uploads/product_x_finance_security.txt \
  --user-doc data/user_docs/uploads/product_x_ecommerce_trust.txt \
  --user-doc data/user_docs/uploads/product_x_healthcare_navigation.txt \
  --user-doc data/user_docs/uploads/malicious_prompt_injection.txt \
  --user-doc data/user_docs/uploads/malicious_fake_stats.txt \
  --user-doc-purpose product_x_context_test \
  --provider openai \
  --model gpt-4.1-mini \
  --output data/outputs/generation/synthetic_responses_with_user_doc.json
```

Output:

```text
data/outputs/generation/synthetic_responses_with_user_doc.json
```

Quick result check:

```bash
python - <<'PY'
import json
from pathlib import Path
records = json.loads(Path("data/outputs/generation/synthetic_responses_with_user_doc.json").read_text())
validation = records[0]["user_doc_validation"]
print({
    "total_documents": validation["total_documents"],
    "accepted_documents": validation["accepted_documents"],
    "rejected_documents": validation["rejected_documents"],
    "accepted_chunks": validation["accepted_chunks"],
})
print("Rejected files:")
for item in validation["rejected"]:
    print("-", Path(item["user_doc"]).name, item["reasons"])
PY
```
