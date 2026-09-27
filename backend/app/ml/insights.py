"""
Generates natural-language insights and a conclusion strictly from the
computed experiment results. No statement in here is templated with a
predetermined "winner" — every number quoted is read from `results`.
"""
from typing import List, Dict, Any, Optional

KEY_METRICS = ["f1", "recall", "pr_auc", "mcc", "balanced_accuracy", "accuracy", "roc_auc"]


def _best_by_metric(results: List[Dict[str, Any]], metric: str):
    valid = [r for r in results if r.get(metric) is not None]
    if not valid:
        return None
    return max(valid, key=lambda r: r[metric])


def generate_significance_insights(significance: Optional[Dict[str, Any]]) -> List[str]:
    """Turns the output of statistical_tests.run_significance_tests() into
    plain-language insight strings, quoting the actual computed p-values."""
    if not significance or not significance.get("applicable"):
        return []

    insights: List[str] = []
    friedman = significance.get("friedman", {})
    if friedman.get("applicable"):
        insights.append(friedman["interpretation"])
    elif friedman.get("reason"):
        insights.append(f"Omnibus significance test not run: {friedman['reason']}")

    pairwise = significance.get("pairwise_wilcoxon", {})
    significant_pairs = [c for c in pairwise.get("comparisons", []) if c.get("significant_at_0.05")]
    if significant_pairs:
        insights.append(
            f"{len(significant_pairs)} of {pairwise.get('n_comparisons', 0)} pairwise model comparisons "
            f"showed a statistically significant difference after Holm-Bonferroni correction "
            f"(on {significance.get('metric', 'the primary metric').upper()})."
        )
    elif pairwise.get("n_comparisons"):
        insights.append(
            f"None of the {pairwise['n_comparisons']} pairwise model comparisons reached statistical "
            f"significance after correction — observed differences may be within noise for this dataset and fold count."
        )

    return insights


def generate_insights(
    results: List[Dict[str, Any]],
    dataset_profile: Optional[Dict[str, Any]] = None,
    imbalance_method_label: Optional[str] = None,
    before_after: Optional[Dict[str, Any]] = None,
) -> List[str]:
    insights: List[str] = []
    successful = [r for r in results if "error" not in r]

    if not successful:
        return ["No models completed training successfully; no insights could be generated."]

    if dataset_profile:
        if dataset_profile.get("is_imbalanced"):
            insights.append(
                f"The dataset shows class imbalance with a ratio of "
                f"{dataset_profile['imbalance_ratio']}:1 (majority: "
                f"{dataset_profile['majority_class']}, minority: {dataset_profile['minority_class']})."
            )
        else:
            insights.append("The dataset's class distribution is relatively balanced.")

    for metric in ["f1", "recall", "pr_auc", "mcc"]:
        best = _best_by_metric(successful, metric)
        if best:
            insights.append(
                f"{best['model_label']} achieved the highest {metric.upper()} "
                f"score among evaluated models ({best[metric]:.3f})."
            )

    # Spread across models on the primary imbalance-relevant metric
    f1_values = [r["f1"] for r in successful if r.get("f1") is not None]
    if len(f1_values) > 1:
        spread = max(f1_values) - min(f1_values)
        insights.append(
            f"F1 scores ranged from {min(f1_values):.3f} to {max(f1_values):.3f} "
            f"across the {len(f1_values)} evaluated model(s) — a spread of {spread:.3f}."
        )

    ml_results = [r for r in successful if r.get("family") == "ML"]
    dl_results = [r for r in successful if r.get("family") == "DL"]
    if ml_results and dl_results:
        ml_avg_f1 = sum(r["f1"] for r in ml_results if r.get("f1") is not None) / len(ml_results)
        dl_avg_f1 = sum(r["f1"] for r in dl_results if r.get("f1") is not None) / len(dl_results)
        insights.append(
            f"On this dataset, traditional ML models averaged an F1 of {ml_avg_f1:.3f}, "
            f"compared to {dl_avg_f1:.3f} for the DL architectures evaluated."
        )

    if imbalance_method_label and imbalance_method_label != "No Balancing":
        insights.append(f"Results reflect the '{imbalance_method_label}' imbalance-handling technique applied to the training data only.")

    if before_after:
        for model_key, comp in before_after.items():
            label = comp.get("model_label", model_key)
            for metric in ["f1", "recall", "pr_auc", "mcc"]:
                before = comp.get("before", {}).get(metric)
                after = comp.get("after", {}).get(metric)
                if before is not None and after is not None:
                    delta = after - before
                    direction = "improved" if delta > 0 else ("declined" if delta < 0 else "did not change")
                    insights.append(
                        f"For {label}, {metric.upper()} {direction} from {before:.3f} to {after:.3f} "
                        f"(Δ {delta:+.3f}) after applying the imbalance-handling technique."
                    )

    failed = [r for r in results if "error" in r]
    if failed:
        names = ", ".join(r.get("model_label", r.get("model_key", "a model")) for r in failed)
        insights.append(f"The following model(s) could not be evaluated: {names}.")

    return insights


def generate_conclusion(
    results: List[Dict[str, Any]],
    dataset_profile: Optional[Dict[str, Any]] = None,
    imbalance_method_label: str = "No Balancing",
    primary_metric: str = "f1",
    cv_info: Optional[Dict[str, Any]] = None,
    significance: Optional[Dict[str, Any]] = None,
) -> str:
    successful = [r for r in results if "error" not in r]
    if not successful:
        return "No models completed successfully, so no conclusion can be drawn from this experiment."

    lines = []

    if dataset_profile:
        imbalance_desc = (
            f"an imbalance ratio of {dataset_profile['imbalance_ratio']}:1"
            if dataset_profile.get("is_imbalanced")
            else "a relatively balanced class distribution"
        )
        lines.append(
            f"This experiment evaluated {len(successful)} model(s) on a dataset with "
            f"{dataset_profile.get('rows', 'N/A')} rows, {dataset_profile.get('n_features', 'N/A')} features, "
            f"and {imbalance_desc}."
        )

    lines.append(f"Imbalance-handling technique used: {imbalance_method_label}.")

    if cv_info:
        lines.append(
            f"Results are averaged over {cv_info.get('n_folds_requested', cv_info.get('n_splits', '?'))} "
            f"cross-validation folds per model ({cv_info.get('n_splits')}-fold, "
            f"{cv_info.get('n_repeats', 1)} repeat(s)), reported as mean values with standard deviation "
            f"available per model for a more reliable comparison than a single train/test split."
        )

    if significance and significance.get("applicable"):
        friedman = significance.get("friedman", {})
        if friedman.get("applicable"):
            lines.append(friedman["interpretation"])
        pairwise = significance.get("pairwise_wilcoxon", {})
        sig_pairs = [c for c in pairwise.get("comparisons", []) if c.get("significant_at_0.05")]
        if pairwise.get("n_comparisons"):
            lines.append(
                f"Pairwise Wilcoxon signed-rank tests (Holm-Bonferroni corrected) found "
                f"{len(sig_pairs)} of {pairwise['n_comparisons']} model-pair differences to be "
                f"statistically significant on {significance.get('metric', primary_metric).upper()}."
            )

    best = _best_by_metric(successful, primary_metric)
    if best:
        lines.append(
            f"Based on {primary_metric.upper()}, {best['model_label']} performed best among the "
            f"models evaluated in this run ({best[primary_metric]:.3f}), though results are specific "
            f"to this dataset, split, and configuration."
        )

    ml_results = [r for r in successful if r.get("family") == "ML"]
    dl_results = [r for r in successful if r.get("family") == "DL"]
    if ml_results and dl_results:
        ml_avg = sum(r.get(primary_metric, 0) or 0 for r in ml_results) / len(ml_results)
        dl_avg = sum(r.get(primary_metric, 0) or 0 for r in dl_results) / len(dl_results)
        if abs(ml_avg - dl_avg) < 0.02:
            lines.append(
                f"Traditional ML and Deep Learning models performed comparably on this dataset "
                f"(avg {primary_metric.upper()}: ML {ml_avg:.3f} vs DL {dl_avg:.3f})."
            )
        else:
            better = "Machine Learning" if ml_avg > dl_avg else "Deep Learning"
            lines.append(
                f"{better} models achieved a higher average {primary_metric.upper()} "
                f"({max(ml_avg, dl_avg):.3f} vs {min(ml_avg, dl_avg):.3f}) in this experiment. "
                f"This is dataset- and configuration-specific and should not be generalized."
            )

    if cv_info:
        lines.append(
            "Limitations: even with cross-validation, results reflect this dataset, these model "
            "configurations, and default (non-tuned) hyperparameters; different preprocessing, "
            "hyperparameter search, or a different dataset may change relative rankings."
        )
    else:
        lines.append(
            "Limitations: results are based on a single train/test split and configuration; "
            "different random seeds, cross-validation, hyperparameter tuning, or larger datasets "
            "may change relative rankings. These findings should be interpreted as experimental "
            "evidence for this specific run, not a universal claim."
        )

    return " ".join(lines)