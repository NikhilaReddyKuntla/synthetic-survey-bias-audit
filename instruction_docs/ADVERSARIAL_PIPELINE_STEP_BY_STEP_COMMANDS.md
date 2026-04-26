# Adversarial Attack + Defense Pipeline (Step-by-Step Commands)

Run all commands from repository root:

```powershell
cd "C:\Users\Nikhila Custom\Documents\MEGA\Career\Portfolio Projects\synthetic-survey-bias-audit"
```

## 1) Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `python` is not recognized, use your installed interpreter path.

## 2) Build/Refresh Clean RAG Store (only if needed)

This creates/refreshes `vector_store/rag_index.faiss` and `vector_store/rag_metadata.json`.

```powershell
python -m src.rag.ingest
python -m src.rag.embed
```

## 3) Generate Attack Documents

```powershell
python -m src.adversarial.generate_attacks
```

Optional domain-specific generation:

```powershell
python -m src.adversarial.generate_attacks --domain finance
```

## 4) Validate Attack Docs with Hardened Defense Gate

This runs static checks + LLM judge and writes defense metadata.

```powershell
python -m src.adversarial.validate_docs `
  --provider groq `
  --model llama-3.1-8b-instant `
  --judge-timeout 20 `
  --judge-min-confidence 0.70 `
  --output data/outputs/adversarial_validation_report.json
```

## 5) Run Main Defended Experiment (Recommended)

This runs clean vs defended conditions and injects only `defense_passed=true` chunks.

```powershell
python -m src.adversarial.run_attack_experiment `
  --questions-file questions.txt `
  --personas data/personas/personas.json `
  --provider groq `
  --model llama-3.1-8b-instant `
  --judge-timeout 20 `
  --judge-min-confidence 0.70
```

Dry-run wiring check (no LLM generation calls):

```powershell
python -m src.adversarial.run_attack_experiment `
  --questions-file questions.txt `
  --personas data/personas/personas.json `
  --dry-run `
  --fast-poison-vectors
```

## 6) Legacy Attack Script (still defense-gated)

Use this only if you want the legacy output layout in `results/`.

```powershell
python -m src.attacks.run_attack `
  --questions-file questions.txt `
  --personas data/personas/personas.json `
  --provider groq `
  --model llama-3.1-8b-instant `
  --judge-timeout 20 `
  --judge-min-confidence 0.70
```

## 7) Validate + Index User Documents (Optional)

High-trust-only policy is enforced before indexing into the user upload store.

```powershell
python -m src.adversarial.upload_validate `
  --input "data\user_docs\uploads\example.txt" `
  --domain finance `
  --provider groq `
  --model llama-3.1-8b-instant `
  --judge-timeout 20 `
  --judge-min-confidence 0.70 `
  --output data/outputs/user_upload_validation_report.json
```

## 8) Generate Survey Responses with `--user-doc` Defense Path (Optional)

```powershell
python -m src.generation.generate_responses `
  --questions-file questions.txt `
  --personas data/personas/personas.json `
  --provider groq `
  --model llama-3.1-8b-instant `
  --user-doc "data\user_docs\uploads\example.txt" `
  --user-doc-purpose finance_context `
  --judge-timeout 20 `
  --judge-min-confidence 0.70 `
  --output data/outputs/synthetic_responses_with_user_doc.json
```

## 9) Run Defense/Guardrail Tests

```powershell
python -m unittest tests/test_defense_decision.py
python -m unittest tests/test_attack_pipeline_guardrails.py
```

## 10) Key Output Files

- `data/outputs/adversarial_validation_report.json`
- `data/outputs/attack_clean.json`
- `data/outputs/attack_poisoned_with_defense.json`
- `data/outputs/attack_analysis.csv`
- `data/outputs/attack_analysis_report.md`
- `results/attack_responses.json` (legacy script)
- `results/attack_responses.csv` (legacy script)
- `results/attack_analysis.csv` (legacy script)
- `data/outputs/user_upload_validation_report.json`
