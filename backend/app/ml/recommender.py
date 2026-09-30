"""
Transparent, rule-based model recommendation engine for imbalanced
classification experiments.

This module intentionally does NOT claim that one model is universally best.
It produces an explainable shortlist from dataset characteristics and the
user's stated priorities.
"""

from __future__ import annotations

from typing import Any, Dict, List


MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "family": "ML",
        "strengths": ["interpretability", "fast", "probabilities", "baseline"],
    },
    "decision_tree": {
        "label": "Decision Tree",
        "family": "ML",
        "strengths": ["interpretability", "nonlinear", "fast"],
    },
    "random_forest": {
        "label": "Random Forest",
        "family": "ML",
        "strengths": ["nonlinear", "robust", "imbalanced", "feature_importance"],
    },
    "svm": {
        "label": "Support Vector Machine",
        "family": "ML",
        "strengths": ["high_dimensional", "nonlinear", "imbalanced"],
    },
    "knn": {
        "label": "K-Nearest Neighbors",
        "family": "ML",
        "strengths": ["local_patterns", "simple"],
    },
    "naive_bayes": {
        "label": "Naive Bayes",
        "family": "ML",
        "strengths": ["fast", "baseline", "high_dimensional"],
    },
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "family": "ML",
        "strengths": ["nonlinear", "tabular", "strong_baseline"],
    },
    "adaboost": {
        "label": "AdaBoost",
        "family": "ML",
        "strengths": ["nonlinear", "ensemble", "tabular"],
    },
    "xgboost": {
        "label": "XGBoost",
        "family": "ML",
        "strengths": ["nonlinear", "tabular", "imbalanced", "strong_baseline"],
    },
    "lightgbm": {
        "label": "LightGBM",
        "family": "ML",
        "strengths": ["nonlinear", "tabular", "large_data", "imbalanced"],
    },
    "ann": {"label": "ANN", "family": "DL", "strengths": ["nonlinear", "tabular"]},
    "mlp": {"label": "MLP", "family": "DL", "strengths": ["nonlinear", "tabular"]},
    "cnn": {"label": "CNN", "family": "DL", "strengths": ["learned_features"]},
    "cnn_1d": {"label": "1D CNN", "family": "DL", "strengths": ["structured_features"]},
    "lstm": {"label": "LSTM", "family": "DL", "strengths": ["sequential_patterns"]},
    "gru": {"label": "GRU", "family": "DL", "strengths": ["sequential_patterns", "faster_sequence"]},
    "bilstm": {"label": "BiLSTM", "family": "DL", "strengths": ["sequential_patterns"]},
    "bigru": {"label": "BiGRU", "family": "DL", "strengths": ["sequential_patterns"]},
    "cnn_lstm": {"label": "CNN-LSTM", "family": "DL", "strengths": ["sequence_features"]},
    "attention_nn": {"label": "Attention NN", "family": "DL", "strengths": ["learned_features", "complex_patterns"]},
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(value: Any, default: str) -> str:
    return str(value or default).strip().lower()


def recommend_models(request: Dict[str, Any]) -> Dict[str, Any]:
    """Return an explainable shortlist and rule-based rationale."""

    rows = int(_num(request.get("rows"), 0))
    features = int(_num(request.get("features"), 0))
    classes = max(2, int(_num(request.get("classes"), 2)))
    imbalance_ratio = _num(request.get("imbalance_ratio"), 1.0)

    goal = _text(request.get("primary_goal"), "balanced_performance")
    family = _text(request.get("model_family"), "both")
    interpretability = _text(request.get("interpretability"), "medium")
    speed = _text(request.get("training_speed"), "balanced")
    probability = _text(request.get("probability_outputs"), "useful")

    candidates = [
        key for key, info in MODEL_INFO.items()
        if family == "both" or info["family"].lower() == family
    ]

    scores: Dict[str, float] = {key: 0.0 for key in candidates}
    reasons: Dict[str, List[str]] = {key: [] for key in candidates}

    strongly_imbalanced = imbalance_ratio >= 5.0
    large_dataset = rows >= 10000
    high_dimensional = features >= 50

    for key in candidates:
        info = MODEL_INFO[key]
        strengths = set(info["strengths"])

        # General tabular classification baseline.
        if key in {"logistic_regression", "random_forest", "gradient_boosting", "xgboost"}:
            scores[key] += 2
            reasons[key].append("Provides a useful baseline for tabular classification.")

        if strongly_imbalanced and "imbalanced" in strengths:
            scores[key] += 3
            reasons[key].append("Supports an imbalance-focused evaluation strategy well; use recall, F1, PR-AUC and MCC rather than accuracy alone.")

        if large_dataset and "large_data" in strengths:
            scores[key] += 3
            reasons[key].append("The dataset size makes scalable tree-based learning relevant.")

        if high_dimensional and "high_dimensional" in strengths:
            scores[key] += 2
            reasons[key].append("The relatively high feature count makes this model a useful high-dimensional candidate.")

        if interpretability == "high" and "interpretability" in strengths:
            scores[key] += 4
            reasons[key].append("Interpretability is a stated priority.")

        if speed == "fast" and key in {"logistic_regression", "decision_tree", "naive_bayes"}:
            scores[key] += 3
            reasons[key].append("Fits a fast-training requirement.")

        if speed == "balanced" and key in {"random_forest", "gradient_boosting", "xgboost", "logistic_regression"}:
            scores[key] += 1

        if probability == "important" and key in {"logistic_regression", "random_forest", "gradient_boosting", "xgboost", "lightgbm", "naive_bayes"}:
            scores[key] += 2
            reasons[key].append("Suitable for probability-based evaluation such as PR-AUC/ROC-AUC when configured appropriately.")

        if goal == "minority_recall" and "imbalanced" in strengths:
            scores[key] += 2
            reasons[key].append("Candidate for experiments where minority-class detection is important.")

        if goal == "interpretability" and "interpretability" in strengths:
            scores[key] += 3

        if goal == "speed" and key in {"logistic_regression", "naive_bayes", "decision_tree"}:
            scores[key] += 3

        if goal == "nonlinear_patterns" and "nonlinear" in strengths:
            scores[key] += 3
            reasons[key].append("Can model nonlinear relationships.")

        if goal == "balanced_performance" and "strong_baseline" in strengths:
            scores[key] += 2

    ranked = sorted(candidates, key=lambda key: (-scores[key], MODEL_INFO[key]["label"]))

    # Keep the shortlist practical and diverse. This is an ordering of the
    # rule-based recommendations, not a claim that the first model is best.
    shortlist_keys = ranked[:5]

    # Ensure an interpretable baseline is present when the user wants one.
    if interpretability in {"high", "medium"} and "logistic_regression" in candidates:
        if "logistic_regression" not in shortlist_keys:
            shortlist_keys[-1] = "logistic_regression"

    # Add a transparent caveat for DL on ordinary tabular datasets.
    caveats: List[str] = []
    if family == "both" and not large_dataset:
        caveats.append("For a smaller tabular dataset, DL models are useful as comparison candidates but may require more tuning and training time than classical ML models.")
    if strongly_imbalanced:
        caveats.append("The dataset is substantially imbalanced; compare minority-class recall, F1, PR-AUC, MCC, balanced accuracy and G-Mean rather than relying on accuracy alone.")
    if probability == "important":
        caveats.append("Probability-based metrics require suitable probability/score outputs from the selected model and configuration.")

    recommendations = []
    for key in shortlist_keys:
        recommendations.append({
            "key": key,
            "label": MODEL_INFO[key]["label"],
            "family": MODEL_INFO[key]["family"],
            "score": round(scores[key], 2),
            "reasons": reasons[key] or ["Included as a transparent comparison candidate."],
        })

    return {
        "engine": "transparent_rule_based",
        "disclaimer": "Recommendations are evidence-guided experiment candidates, not a claim that one model will perform best before evaluation.",
        "dataset_summary": {
            "rows": rows,
            "features": features,
            "classes": classes,
            "imbalance_ratio": imbalance_ratio,
        },
        "criteria": {
            "primary_goal": goal,
            "model_family": family,
            "interpretability": interpretability,
            "training_speed": speed,
            "probability_outputs": probability,
        },
        "recommendations": recommendations,
        "caveats": caveats,
        "all_candidate_keys": ranked,
    }
