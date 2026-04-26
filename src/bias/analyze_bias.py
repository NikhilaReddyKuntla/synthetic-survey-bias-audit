from __future__ import annotations
import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from scipy.stats import chisquare, entropy


# ── Helper Functions ─────────────────────────────────────
def clean_label(val):
    return re.sub(r"\s+", " ", str(val).replace("\xa0", "")).strip()


def get_percent(df, label, percent_col="United States!!Percent"):
    row = df[df['clean_label'] == label]
    if row.empty:
        return 0.0
    for col in df.columns:
        if percent_col in col and 'Error' not in col:
            val = str(row[col].values[0]).replace('%','').replace(',','').strip()
            try:
                return float(val)
            except:
                return 0.0
    return 0.0


def get_estimate(df, label):
    row = df[df['clean_label'] == label]
    if row.empty:
        return 0.0
    val = str(row.iloc[0, 1]).replace(',','').strip()
    try:
        return float(val)
    except:
        return 0.0


def get_income_pct(df, label):
    row = df[df['clean_label'] == label]
    if row.empty:
        return 0.0
    val = str(row['United States!!Households!!Estimate'].values[0]).replace('%','').strip()
    try:
        return float(val)
    except:
        return 0.0


def get_edu_pct(df, label):
    row = df[df['clean_label'] == label]
    if row.empty:
        return 0.0
    for col in df.columns:
        if 'Percent' in col and 'Error' not in col and 'Total' in col:
            val = str(row[col].values[0]).replace('%','').strip()
            try:
                return float(val)
            except:
                return 0.0
    return 0.0


# ── Load ACS Files ──────────────────────────────────────
acs_dir = Path("data/raw_sources/acs")
demos  = pd.read_csv(acs_dir / "demographics.csv")
income = pd.read_csv(acs_dir / "income.csv")
edu    = pd.read_csv(acs_dir / "education.csv")
econ   = pd.read_csv(acs_dir / "economic_characteristics.csv")

for df in [demos, income, edu, econ]:
    df['clean_label'] = df['Label (Grouping)'].apply(clean_label)


# ── ACS Race Distribution ────────────────────────────────
acs_race = {
    "White":                     get_percent(demos, "White alone"),
    "Hispanic or Latino":        get_percent(demos, "Hispanic or Latino (of any race)"),
    "Black or African American": get_percent(demos, "Black or African American alone"),
    "Asian":                     get_percent(demos, "Asian alone"),
    "Other":                     round(
        100
        - get_percent(demos, "White alone")
        - get_percent(demos, "Hispanic or Latino (of any race)")
        - get_percent(demos, "Black or African American alone")
        - get_percent(demos, "Asian alone"),
        1
    ),
}

# ── ACS Gender Distribution ──────────────────────────────
acs_gender = {
    "Male":   get_percent(demos, "Male"),
    "Female": get_percent(demos, "Female"),
}

# ── ACS Age Distribution ─────────────────────────────────
acs_age_raw = {
    "18-29": get_percent(demos, "20 to 24 years"),
    "30-44": get_percent(demos, "25 to 34 years") + get_percent(demos, "35 to 44 years"),
    "45-64": (get_percent(demos, "45 to 54 years") +
              get_percent(demos, "55 to 59 years") +
              get_percent(demos, "60 to 64 years")),
    "65+":   (get_percent(demos, "65 to 74 years") +
              get_percent(demos, "75 to 84 years") +
              get_percent(demos, "85 years and over")),
}
total_age = sum(acs_age_raw.values())
acs_age = {k: round(v / total_age * 100, 1) for k, v in acs_age_raw.items()}

# ── ACS Income Distribution ──────────────────────────────
acs_income_raw = {
    "Less than $25k": (get_income_pct(income, "Less than $10,000") +
                       get_income_pct(income, "$10,000 to $14,999") +
                       get_income_pct(income, "$15,000 to $24,999")),
    "$25k-$49k":      (get_income_pct(income, "$25,000 to $34,999") +
                       get_income_pct(income, "$35,000 to $49,999")),
    "$50k-$74k":       get_income_pct(income, "$50,000 to $74,999"),
    "$75k-$99k":       get_income_pct(income, "$75,000 to $99,999"),
    "$100k or more":  (get_income_pct(income, "$100,000 to $149,999") +
                       get_income_pct(income, "$150,000 to $199,999") +
                       get_income_pct(income, "$200,000 or more")),
}
total_inc = sum(acs_income_raw.values())
acs_income = {k: round(v / total_inc * 100, 1) for k, v in acs_income_raw.items()}

# ── ACS Education Distribution ───────────────────────────
acs_edu_raw = {
    "Less than high school": (get_edu_pct(edu, "Less than 9th grade") +
                              get_edu_pct(edu, "9th to 12th grade, no diploma")),
    "High school graduate":   get_edu_pct(edu, "High school graduate (includes equivalency)"),
    "Some college":           (get_edu_pct(edu, "Some college, no degree") +
                               get_edu_pct(edu, "Associate's degree")),
    "Bachelor's degree":      get_edu_pct(edu, "Bachelor's degree"),
    "Graduate degree":        get_edu_pct(edu, "Graduate or professional degree"),
}
total_edu = sum(acs_edu_raw.values())
if total_edu == 0:
    acs_edu = {k: 20.0 for k in acs_edu_raw}
else:
    acs_edu = {k: round(v / total_edu * 100, 1) for k, v in acs_edu_raw.items()}

# ── ACS Employment Distribution ──────────────────────────
emp_employed   = get_estimate(econ, "Employed")
emp_unemployed = get_estimate(econ, "Unemployed")
emp_not_in     = get_estimate(econ, "Not in labor force")
emp_total      = emp_employed + emp_unemployed + emp_not_in
acs_employment = {
    "Employed":           round(emp_employed   / emp_total * 100, 1),
    "Unemployed":         round(emp_unemployed / emp_total * 100, 1),
    "Not in labor force": round(emp_not_in     / emp_total * 100, 1),
}

print("=== ACS REAL US DISTRIBUTIONS ===")
print(f"Race:       {acs_race}")
print(f"Gender:     {acs_gender}")
print(f"Age:        {acs_age}")
print(f"Income:     {acs_income}")
print(f"Education:  {acs_edu}")
print(f"Employment: {acs_employment}")


# ── Load Synthetic Survey Data ───────────────────────────
gpt = pd.read_csv("data/outputs/gpt_synthetic_survey.csv")
ds  = pd.read_csv("data/outputs/deepseek_synthetic_survey.csv")
print(f"\nGPT records: {len(gpt)}")
print(f"DeepSeek records: {len(ds)}")


# ── Analysis Function ────────────────────────────────────
def analyze(model_df, acs_dist, col, model_name):
    categories = list(acs_dist.keys())
    acs_pct = np.array(list(acs_dist.values()), dtype=float)
    acs_pct = acs_pct / acs_pct.sum() * 100

    counts = model_df[col].value_counts()
    model_pct = np.array([counts.get(c, 0) for c in categories], dtype=float)
    total = model_pct.sum()
    if total == 0:
        print(f"  No data for {col}")
        return None
    model_pct_norm = model_pct / total * 100

    acs_d = (acs_pct + 1e-9) / (acs_pct + 1e-9).sum()
    mod_d = (model_pct_norm + 1e-9) / (model_pct_norm + 1e-9).sum()
    kl = entropy(mod_d, acs_d)

    expected = acs_pct / 100 * total
    try:
        chi2, p = chisquare(model_pct + 0.5, expected + 0.5)
    except Exception:
        chi2, p = 0.0, 1.0

    print(f"\n--- {model_name} vs ACS: {col} ---")
    print(f"{'Category':<30} {'ACS%':>8} {'Model%':>8} {'Gap':>8}")
    print("-" * 58)
    for cat, ap, mp in zip(categories, acs_pct, model_pct_norm):
        print(f"{cat:<30} {ap:>8.1f} {mp:>8.1f} {mp-ap:>+8.1f}")
    print(f"KL Divergence: {kl:.4f}")
    print(f"Chi-square p={p:.4f} → {'⚠️  BIASED' if p < 0.05 else '✅ OK'}")

    return {
        "kl_divergence": round(kl, 4),
        "chi2":          round(chi2, 4),
        "p_value":       round(p, 4),
        "biased":        bool(p < 0.05),
    }


# ── Run All Dimensions ───────────────────────────────────
dims = {
    "race":       ("race_ethnicity",    acs_race),
    "gender":     ("gender",            acs_gender),
    "age":        ("age_group",         acs_age),
    "income":     ("income_bracket",    acs_income),
    "education":  ("education",         acs_edu),
    "employment": ("employment_status", acs_employment),
}

print("\n\n========================================")
print("GPT ANALYSIS")
print("========================================")
gpt_results = {dim: analyze(gpt, dist, col, "GPT")
               for dim, (col, dist) in dims.items()}

print("\n\n========================================")
print("DEEPSEEK ANALYSIS")
print("========================================")
ds_results = {dim: analyze(ds, dist, col, "DeepSeek")
              for dim, (col, dist) in dims.items()}


# ── Response Score Bias by Race ──────────────────────────
print("\n\n========================================")
print("RESPONSE SCORE BIAS BY RACE")
print("(1=comfortable, 4=struggling)")
print("========================================")
print("\nGPT avg response score by race:")
print(gpt.groupby('race_ethnicity')['response_score'].mean().round(2)
        .sort_values(ascending=False))
print("\nDeepSeek avg response score by race:")
print(ds.groupby('race_ethnicity')['response_score'].mean().round(2)
        .sort_values(ascending=False))


# ── Summary Table ────────────────────────────────────────
print("\n\n========================================")
print("OVERALL BIAS SUMMARY")
print("========================================")
print(f"\n{'Dimension':<15} {'GPT KL':>10} {'DS KL':>10} {'GPT Biased':>12} {'DS Biased':>12}")
print("-" * 62)
for dim in dims:
    gr = gpt_results.get(dim) or {}
    dr = ds_results.get(dim)  or {}
    print(f"{dim:<15} "
          f"{gr.get('kl_divergence', 'N/A'):>10} "
          f"{dr.get('kl_divergence', 'N/A'):>10} "
          f"{'YES' if gr.get('biased') else 'NO':>12} "
          f"{'YES' if dr.get('biased') else 'NO':>12}")


# ── Save Report ──────────────────────────────────────────
report = {
    "benchmark_source": "ACS 2024 US Census",
    "question": "How would you describe your current financial situation?",
    "acs_distributions": {
        "race": acs_race, "gender": acs_gender, "age": acs_age,
        "income": acs_income, "education": acs_edu, "employment": acs_employment,
    },
    "gpt":      {"records": len(gpt),  "results": gpt_results},
    "deepseek": {"records": len(ds),   "results": ds_results},
    "response_score_by_race": {
        "gpt":      gpt.groupby('race_ethnicity')['response_score'].mean().round(2).to_dict(),
        "deepseek": ds.groupby('race_ethnicity')['response_score'].mean().round(2).to_dict(),
    },
    "conclusion": (
        "GPT and DeepSeek synthetic surveys compared against ACS 2024 real US "
        "demographic distributions. Models show demographic bias when their "
        "generated distributions deviate significantly from census benchmarks."
    ),
}

Path("data/outputs").mkdir(parents=True, exist_ok=True)
with open("data/outputs/model_comparison_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("\n✅ Report saved → data/outputs/model_comparison_report.json")