# synthetic-survey-bias-audit
Auditing bias and adversarial vulnerability in LLM-generated synthetic surveys using persona prompting, statistical evaluation, and RAG-based defenses.

---

## 📌 Overview

Companies are increasingly using Large Language Models (LLMs) to simulate customer surveys by prompting models to act as demographic personas. While this approach is faster and cheaper than traditional surveys, it introduces two major risks:

1. **Demographic Bias** – LLM-generated personas may overrepresent majority groups and misrepresent minority populations.
2. **Adversarial Manipulation** – Retrieval-Augmented Generation (RAG) pipelines can be exploited by injecting misleading “market context” to skew survey results.

This project builds a **two-part auditing framework** to detect, quantify, and defend against these risks before they lead to incorrect business decisions.

---

## 🎯 Objectives

- Detect systematic **demographic bias** in LLM-generated survey responses
- Evaluate **robustness against adversarial attacks** (e.g., PoisonedRAG-style injections)
- Compare multiple models on bias and vulnerability
- Build a lightweight **defense layer** to reduce manipulation

---

## 🧱 System Architecture

### 1. RAG-based Survey System
- Uses retrieval to provide contextual “market data”
- Generates persona-based survey responses using LLMs

### 2. Bias Detection Module
- Simulates responses across demographic personas
- Compares outputs with real-world benchmarks (e.g., Pew/YouGov)
- Uses statistical measures (KL divergence, proportion tests)

### 3. Adversarial Attack + Defense
- Injects manipulated documents into the RAG pipeline
- Measures response shifts due to adversarial context
- Applies cross-checking with a clean knowledge base to detect manipulation

---

## 🛠️ Tech Stack

- **LLMs**: GPT-3.5 Turbo, LLaMA 3, Mixtral, Mistral 7B
- **Retrieval**: Sentence-Transformers
- **Frameworks**: Python, LangChain (optional)
- **Data Sources**: Pew Research, YouGov public datasets
- **Evaluation**: NumPy, SciPy, Pandas

---

## 📊 Evaluation Metrics

- **Bias Detection**
  - Statistical divergence from real-world survey data
  - Bias measured across demographics (age, income, ethnicity, etc.)

- **Adversarial Robustness**
  - % change in responses after adversarial injection (>10% = significant)
  - Reduction in shift after applying defense layer

- **Model Comparison**
  - Ranking models by bias magnitude and robustness

---

## 🧪 Baseline

A naive system:
- No persona prompting
- No retrieval augmentation
- No adversarial defense

Used to compare improvements from the proposed framework.

---






## Adversarial Attack + Defense (Part 2)

This repository now includes a lightweight PoisonedRAG simulation and a retrieval-consistency defense layer.

### 1) Run Attack Simulation

```bash
python -m src.attacks.run_attack \
  --questions-file data/questions/questions.txt \
  --personas data/personas/personas.json \
  --domain ecommerce \
  --top-k 3 \
  --provider local \
  --model llama3.1
```

You can add `--dry-run` to validate pipeline wiring without calling an LLM.
For fully offline smoke tests, add `--fast-poison-vectors` to avoid sentence-transformer downloads for injected docs.

Attack outputs:
- `results/attack_responses.json`
- `results/attack_responses.csv`
- `results/attack_analysis.csv`
- `results/poisoned_vector_store/rag_index.faiss`
- `results/poisoned_vector_store/rag_metadata.json`

`attack_success = true` when semantic shift is greater than 10% by default.

### 2) Run Defense

```bash
python -m src.defense.defense \
  --attack-records results/attack_responses.json \
  --provider local \
  --model llama3.1
```

You can add `--dry-run` to run retrieval/filter checks without model calls.

Defense outputs:
- `results/defense_results.json`
- `results/defense_results.csv`

Defense behavior:
- Re-run with retrieved (potentially poisoned) context.
- Re-run with context filtered against trusted clean chunk IDs.
- Flag record as suspicious when untrusted chunks appear or divergence exceeds threshold.
- Use filtered response as defended output for suspicious records.
