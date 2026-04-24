import pandas as pd
from scipy.stats import entropy

def flag_biased_groups(ztest_results, threshold=0.05):
    """Flag demographic groups with statistically significant bias"""
    return ztest_results[ztest_results['p_value'] < threshold]

def compute_bias_summary(kl_scores):
    """Summarize bias scores across demographic groups"""
    import pandas as pd
    scores = pd.Series(kl_scores)
    return {
        'most_biased_group':  scores.idxmax(),
        'least_biased_group': scores.idxmin(),
        'mean_kl': round(scores.mean(), 4),
        'max_kl':  round(scores.max(), 4),
        'min_kl':  round(scores.min(), 4)
    }