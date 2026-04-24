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

---
## 🔍 Bias Detection Module

**Built by:** Shravani Prakash Maskar

### What It Does
Detects systematic demographic bias in LLM-generated survey responses 
by comparing them against real-world survey data from Pew Research.

### Data Sources
- **Pew Research ATP Wave 157** (October 2024) — 5,155 US workers
  - Questions about AI attitudes, job satisfaction, remote work
  - Ground truth for bias comparison
- **SHED 2024** (Federal Reserve) — 12,295 respondents
  - Financial wellbeing survey
  - Saved for future financial bias analysis

### Approach
1. Built 72 demographic personas (race × age × gender × income)
2. Asked GPT-3.5 the same survey question for each persona
3. Compared LLM responses to real Pew distributions
4. Computed KL divergence and z-tests to quantify bias

### Key Findings
| Race | KL Divergence | Significant? |
|------|--------------|--------------|
| White | 0.2376 | No (p=0.71) |
| Asian | 0.4395 | No (p=0.60) |
| Hispanic | 0.5381 | No (p=0.15) |
| **Black** | **0.7811** | **Yes (p=0.005)** |

- LLM is **3x more biased** against Black personas vs White
- Without persona prompting, LLM defaults to pessimistic response 100% of the time
- White personas are the **only group** that shifts away from the pessimistic default
- Black personas require **higher income** to receive same optimism as White personas

### Baseline
Without persona prompting, the LLM always responds with 3
("AI will hurt more workers than it helps") — showing the
bias is in how the LLM interprets race, not just a default answer.

### How to Run
```bash
pip install -r requirements.txt
```
Then run notebooks in order:
1. `notebooks/01_load_and_explore.ipynb`
2. `notebooks/02_persona_prompting.ipynb`
3. `notebooks/03_bias_metrics.ipynb`

### Output Files
- `data/processed/pew_clean.csv` — cleaned Pew data
- `data/processed/llm_responses.csv` — synthetic persona responses
- `data/processed/bias_by_race.png` — KL divergence chart
- `data/processed/bias_heatmap.png` — race × income heatmap
- `data/processed/ztest_results.csv` — statistical results

### Using the src/ Modules
The bias detection logic is also available as importable Python modules:

```python
# Load and clean data
from src.bias.detect import compute_kl_divergence, compute_ztest
from src.persona.generate_personas import build_personas, build_prompt

# Generate personas
personas = build_personas()  # returns 72 combinations

# Compute bias
kl_scores = compute_kl_divergence(pew, llm, ['White','Black','Hispanic','Asian'])
ztest_results = compute_ztest(pew, llm, ['White','Black','Hispanic','Asian'])
```



