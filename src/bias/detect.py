import numpy as np
import pandas as pd
from scipy.stats import entropy
from statsmodels.stats.proportion import proportions_ztest

RESPONSES = [1.0, 2.0, 3.0, 4.0]

def get_distribution(df, group_col, group_val, response_col):
    subset = df[df[group_col] == group_val][response_col].dropna()
    counts = subset.value_counts()
    dist = np.array([counts.get(r, 0) for r in RESPONSES], dtype=float)
    dist = (dist + 1e-9) / dist.sum()
    return dist

def compute_kl_divergence(pew, llm, races):
    scores = {}
    for race in races:
        real  = get_distribution(pew, 'race', race, 'AIWRKOPPO_W157')
        llm_d = get_distribution(llm, 'race', race, 'llm_response')
        scores[race] = round(entropy(llm_d, real), 4)
    return scores

def compute_ztest(pew, llm, races):
    results = []
    for race in races:
        pew_race  = pew[pew['race'] == race]['AIWRKOPPO_W157'].dropna()
        pew_hurts = (pew_race >= 3).sum()
        llm_race  = llm[llm['race'] == race]['llm_response'].dropna()
        llm_hurts = (llm_race >= 3).sum()
        count = np.array([llm_hurts, pew_hurts])
        nobs  = np.array([len(llm_race), len(pew_race)])
        z, p  = proportions_ztest(count, nobs)
        results.append({
            'race': race,
            'pew_pct_hurt': round(pew_hurts/len(pew_race)*100, 1),
            'llm_pct_hurt': round(llm_hurts/len(llm_race)*100, 1),
            'z_score': round(z, 3),
            'p_value': round(p, 4),
            'significant': p < 0.05
        })
    return pd.DataFrame(results)