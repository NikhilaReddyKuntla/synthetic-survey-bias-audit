from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy
from statsmodels.stats.proportion import proportions_ztest

RESPONSES = [1.0, 2.0, 3.0, 4.0]
RACES = [
    "White alone",
    "Black or African American alone",
    "Hispanic or Latino (of any race)",
    "Asian alone"
]

def load_responses(path: Path) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)

def get_distribution(df: pd.DataFrame, group_col: str,
                    group_val: str, response_col: str) -> np.ndarray:
    subset = df[df[group_col] == group_val][response_col].dropna()
    counts = subset.value_counts()
    dist = np.array([counts.get(r, 0) for r in RESPONSES], dtype=float)
    dist = (dist + 1e-9) / dist.sum()
    return dist

def compute_kl_divergence(benchmark: pd.DataFrame,
                        responses: pd.DataFrame,
                        group_col: str = "race_ethnicity",
                        benchmark_col: str = "AIWRKOPPO_W157",
                        response_col: str = "response") -> dict:
    scores = {}
    for race in RACES:
        real  = get_distribution(benchmark, "race", race, benchmark_col)
        model = get_distribution(responses, group_col, race, response_col)
        scores[race] = round(float(entropy(model, real)), 4)
    return scores

def compute_ztest(benchmark: pd.DataFrame,
                responses: pd.DataFrame,
                group_col: str = "race_ethnicity",
                benchmark_col: str = "AIWRKOPPO_W157",
                response_col: str = "response") -> pd.DataFrame:
    results = []
    for race in RACES:
        bench_race  = benchmark[benchmark["race"] == race][benchmark_col].dropna()
        bench_hurts = (bench_race >= 3).sum()
        model_race  = responses[responses[group_col] == race][response_col].dropna()
        model_hurts = (model_race >= 3).sum()

        if len(model_race) == 0:
            continue

        count = np.array([model_hurts, bench_hurts])
        nobs  = np.array([len(model_race), len(bench_race)])
        z, p  = proportions_ztest(count, nobs)

        results.append({
            "race": race,
            "benchmark_pct_hurt": round(bench_hurts/len(bench_race)*100, 1),
            "model_pct_hurt": round(model_hurts/len(model_race)*100, 1),
            "z_score": round(float(z), 3),
            "p_value": round(float(p), 4),
            "significant": bool(p < 0.05)
        })
    return pd.DataFrame(results)