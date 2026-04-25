from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.bias.detect import compute_kl_divergence, compute_ztest, load_responses
from src.utils.helpers import data_dir


def evaluate_model(responses_path: Path,
                   benchmark: pd.DataFrame,
                   model_name: str) -> dict:
    """Full bias evaluation for one model"""
    responses = load_responses(responses_path)

    kl_scores = compute_kl_divergence(benchmark, responses)
    ztests    = compute_ztest(benchmark, responses)

    return {
        "model": model_name,
        "kl_divergence": kl_scores,
        "mean_kl": round(float(np.mean(list(kl_scores.values()))), 4),
        "most_biased_group": max(kl_scores, key=kl_scores.get),
        "least_biased_group": min(kl_scores, key=kl_scores.get),
        "ztest_results": ztests.to_dict(orient="records")
    }


def compare_models(benchmark: pd.DataFrame,
                   outputs_dir: Path) -> dict:
    """Compare bias scores across GPT, Groq, and RAG"""
    results = []
    models = {
        "GPT":  outputs_dir / "gpt_responses.json",
        "Groq": outputs_dir / "groq_responses.json",
        "RAG":  outputs_dir / "our_rag_responses.json",
    }

    for name, path in models.items():
        if path.exists():
            result = evaluate_model(path, benchmark, name)
            results.append(result)
            print(f"✅ {name}: mean KL={result['mean_kl']} "
                  f"most biased={result['most_biased_group']}")
        else:
            print(f"⚠️  {name}: file not found → {path}")

    ranked = sorted(results, key=lambda x: x["mean_kl"])

    return {
        "model_results": results,
        "ranking": [r["model"] for r in ranked],
        "conclusion": f"{ranked[0]['model']} shows least bias "
                      f"(mean KL={ranked[0]['mean_kl']})" if ranked else "No results"
    }


def save_report(report: dict, outputs_dir: Path) -> None:
    """Save comparison as JSON and markdown"""
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_path = outputs_dir / "model_comparison_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Saved: {json_path}")

    # Markdown summary
    md_path = outputs_dir / "model_comparison_summary.md"
    with open(md_path, "w") as f:
        f.write("# Model Comparison Summary\n\n")
        f.write("## Ranking (least to most biased)\n")
        for i, model in enumerate(report["ranking"], 1):
            f.write(f"{i}. {model}\n")
        f.write(f"\n## Conclusion\n{report['conclusion']}\n\n")
        f.write("## Detailed Results\n\n")
        for result in report["model_results"]:
            f.write(f"### {result['model']}\n")
            f.write(f"- Mean KL Divergence: {result['mean_kl']}\n")
            f.write(f"- Most biased group: {result['most_biased_group']}\n")
            f.write(f"- Least biased group: {result['least_biased_group']}\n\n")
    print(f"✅ Saved: {md_path}")


def main() -> None:
    import pandas as pd
    outputs = data_dir() / "outputs"
    benchmark = pd.read_csv(data_dir() / "processed" / "pew_clean.csv")
    report = compare_models(benchmark, outputs)
    save_report(report, outputs)


if __name__ == "__main__":
    main()