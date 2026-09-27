"""
Statistical significance testing across cross-validated models, following the
methodology recommended by Demsar (2006), "Statistical Comparisons of
Classifiers over Multiple Data Sets", JMLR 7:1-30 — a standard citation for
this exact procedure in ML thesis work:

  1. Friedman test: an omnibus, non-parametric test for whether 3+ classifiers'
     ranks differ significantly across folds.
  2. Pairwise Wilcoxon signed-rank tests on paired per-fold scores, with
     Holm-Bonferroni correction to control the family-wise error rate across
     multiple pairwise comparisons.

All p-values come from `scipy.stats`; nothing here is approximated or invented.
"""
from typing import List, Dict, Any
from itertools import combinations
import numpy as np
from scipy import stats


def friedman_test(model_fold_scores: Dict[str, List[float]]) -> Dict[str, Any]:
    """model_fold_scores: {model_label: [score_fold_1, score_fold_2, ...]}"""
    labels = list(model_fold_scores.keys())
    if len(labels) < 3:
        return {
            "applicable": False,
            "reason": "The Friedman test requires at least 3 models with completed "
                      "cross-validation; fewer were available in this experiment.",
        }

    lengths = {len(v) for v in model_fold_scores.values()}
    if len(lengths) != 1 or min(lengths) < 2:
        return {
            "applicable": False,
            "reason": "Models must have the same number of completed folds to run the Friedman test.",
        }

    matrix = [model_fold_scores[l] for l in labels]
    try:
        statistic, p_value = stats.friedmanchisquare(*matrix)
    except ValueError as exc:
        return {"applicable": False, "reason": f"Could not compute the Friedman test: {exc}"}

    significant = bool(p_value < 0.05)
    return {
        "applicable": True,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_0.05": significant,
        "models_compared": labels,
        "interpretation": (
            f"The Friedman test {'found' if significant else 'did not find'} a statistically "
            f"significant difference among the {len(labels)} models across folds "
            f"(χ² = {statistic:.3f}, p = {p_value:.4f})."
        ),
    }


def pairwise_wilcoxon(model_fold_scores: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Pairwise Wilcoxon signed-rank test between every pair of models on the same
    folds, with Holm-Bonferroni correction across all pairwise comparisons.
    """
    labels = list(model_fold_scores.keys())
    raw_results = []

    for a, b in combinations(labels, 2):
        scores_a = np.array(model_fold_scores[a])
        scores_b = np.array(model_fold_scores[b])
        if len(scores_a) != len(scores_b) or len(scores_a) < 2:
            continue

        diff = scores_a - scores_b
        if np.all(diff == 0):
            statistic, p_value = 0.0, 1.0
        else:
            try:
                statistic, p_value = stats.wilcoxon(scores_a, scores_b)
            except ValueError:
                continue

        raw_results.append({
            "model_a": a, "model_b": b,
            "mean_a": float(scores_a.mean()), "mean_b": float(scores_b.mean()),
            "statistic": float(statistic), "p_value_raw": float(p_value),
        })

    # Holm-Bonferroni step-down correction, applied across all pairwise tests
    m = len(raw_results)
    ordered = sorted(raw_results, key=lambda r: r["p_value_raw"])
    running_max = 0.0
    for rank, r in enumerate(ordered):
        adjusted = min((m - rank) * r["p_value_raw"], 1.0)
        running_max = max(running_max, adjusted)
        r["p_value_holm"] = running_max
        r["significant_at_0.05"] = bool(running_max < 0.05)
        better = r["model_a"] if r["mean_a"] > r["mean_b"] else r["model_b"]
        r["interpretation"] = (
            f"{r['model_a']} vs {r['model_b']}: "
            f"{'a statistically significant' if r['significant_at_0.05'] else 'no statistically significant'} "
            f"difference (Holm-adjusted p = {r['p_value_holm']:.4f}); {better} scored higher on average."
        )

    return {"n_comparisons": m, "comparisons": ordered}


def run_significance_tests(cv_results: List[Dict[str, Any]], metric: str = "f1") -> Dict[str, Any]:
    """
    cv_results: the list returned by cross_validate_experiment(). Only models
    that completed cross-validation successfully are included.
    """
    usable = {}
    for r in cv_results:
        if "error" in r:
            continue
        details = r.get("cv_details", {}).get("metrics", {}).get(metric)
        if details and len(details.get("values", [])) >= 2:
            usable[r["model_label"]] = details["values"]

    if len(usable) < 2:
        return {
            "applicable": False,
            "reason": "At least 2 successfully cross-validated models are needed for significance testing.",
        }

    friedman = friedman_test(usable)
    pairwise = pairwise_wilcoxon(usable)

    return {"applicable": True, "metric": metric, "friedman": friedman, "pairwise_wilcoxon": pairwise}