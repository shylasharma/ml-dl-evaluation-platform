"""
Research-oriented analysis utilities.

This module does not train models.
It analyzes already-computed experiment results and produces
research-friendly summaries.
"""

from typing import Any, Dict, List, Optional


RESEARCH_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "mcc",
    "balanced_accuracy",
    "g_mean",
]


def _safe_float(value: Any) -> Optional[float]:
    """Convert a metric value to float safely."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_metric_delta(before: Any, after: Any) -> Dict[str, Optional[float]]:
    """
    Calculate absolute and percentage change between two metric values.
    """

    before_value = _safe_float(before)
    after_value = _safe_float(after)

    if before_value is None or after_value is None:
        return {
            "before": before_value,
            "after": after_value,
            "delta": None,
            "percentage_change": None,
        }

    delta = after_value - before_value

    if before_value == 0:
        percentage_change = None
    else:
        percentage_change = (delta / abs(before_value)) * 100

    return {
        "before": round(before_value, 6),
        "after": round(after_value, 6),
        "delta": round(delta, 6),
        "percentage_change": round(percentage_change, 4)
        if percentage_change is not None
        else None,
    }


def analyze_before_after(before: Dict[str, Any],
                         after: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare metrics before and after an imbalance-handling technique.
    """

    comparison = {}

    for metric in RESEARCH_METRICS:
        comparison[metric] = calculate_metric_delta(
            before.get(metric),
            after.get(metric),
        )

    valid_deltas = [
        item["delta"]
        for item in comparison.values()
        if item["delta"] is not None
    ]

    positive_changes = sum(1 for value in valid_deltas if value > 0)
    negative_changes = sum(1 for value in valid_deltas if value < 0)
    unchanged = sum(1 for value in valid_deltas if value == 0)

    return {
        "metrics": comparison,
        "summary": {
            "metrics_improved": positive_changes,
            "metrics_declined": negative_changes,
            "metrics_unchanged": unchanged,
            "metrics_evaluated": len(valid_deltas),
        },
    }


def build_before_after_analysis(
    before_after: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build research analysis for every model in a before/after experiment.
    """

    if not before_after:
        return {
            "available": False,
            "models": {},
        }

    models = {}

    for model_key, data in before_after.items():
        before = data.get("before") or {}
        after = data.get("after") or {}

        if not before or not after:
            models[model_key] = {
                "model_label": data.get("model_label", model_key),
                "available": False,
                "comparison": {},
            }
            continue

        models[model_key] = {
            "model_label": data.get("model_label", model_key),
            "available": True,
            "comparison": analyze_before_after(before, after),
        }

    return {
        "available": bool(models),
        "models": models,
    }


def build_model_comparison(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert model results into a compact research comparison table.
    """

    successful = [
        result
        for result in results
        if "error" not in result
    ]

    comparison = []

    for result in successful:
        row = {
            "model_key": result.get("model_key"),
            "model_label": result.get("model_label"),
            "family": result.get("family"),
            "training_time_sec": result.get("training_time_sec"),
        }

        for metric in RESEARCH_METRICS:
            row[metric] = result.get(metric)

        comparison.append(row)

    return {
        "models_evaluated": len(comparison),
        "models": comparison,
    }


def build_metric_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Organize results metric-by-metric.

    No ranking is imposed here; the purpose is to make the
    metric values easy to compare.
    """

    summary = {}

    for metric in RESEARCH_METRICS:
        values = []

        for result in results:
            value = _safe_float(result.get(metric))

            if value is not None:
                values.append({
                    "model_key": result.get("model_key"),
                    "model_label": result.get("model_label"),
                    "family": result.get("family"),
                    "value": round(value, 6),
                })

        summary[metric] = values

    return summary


def build_research_summary(
    results: List[Dict[str, Any]],
    before_after: Optional[Dict[str, Any]] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
    imbalance_method: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create one research-oriented summary object for an experiment.
    """

    successful_results = [
        result for result in results
        if "error" not in result
    ]

    failed_results = [
        result for result in results
        if "error" in result
    ]

    return {
        "experiment_summary": {
            "models_requested": len(results),
            "models_succeeded": len(successful_results),
            "models_failed": len(failed_results),
            "imbalance_method": imbalance_method,
            "random_state": random_state,
        },

        "dataset": {
    "rows": dataset_profile.get("rows")
    if dataset_profile else None,

    "features": dataset_profile.get("n_features")
    if dataset_profile else None,

    "imbalance_ratio": dataset_profile.get("imbalance_ratio")
    if dataset_profile else None,
},
        "model_comparison": build_model_comparison(
            successful_results
        ),

        "metric_summary": build_metric_summary(
            successful_results
        ),

        "before_after": build_before_after_analysis(
            before_after
        ),
    }