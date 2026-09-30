"""
ML/DL Model Evaluation & Comparison Platform
Hybrid Model Training Utilities

Supported hybridization strategies:
- hard_voting
- soft_voting
- stacking
- blending

Hybridization is intentionally implemented for the existing ML registry.
Deep-learning models are kept separate because they do not share the same
scikit-learn estimator interface used by VotingClassifier/StackingClassifier.

Leakage-safety:
- The hybrid model receives only the already-prepared training data.
- Test/validation data is used only for final evaluation.
- In cross-validation, this module should be called separately for each fold.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from app.ml.metrics import compute_metrics
from app.ml.registry import ML_MODELS


class HybridizationError(Exception):
    """Raised when a hybrid model cannot be constructed or trained."""


HYBRID_METHODS = {
    "none": "No hybridization",
    "hard_voting": "Hard Voting",
    "soft_voting": "Soft Voting",
    "stacking": "Stacking",
    "blending": "Blending",
}


def resolve_hybridization_config(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalize hybridization configuration.

    Expected shape:

    {
        "enabled": True,
        "method": "soft_voting",
        "base_models": ["logistic_regression", "random_forest", "svm"],
        "weights": [1, 2, 1]
    }
    """
    cfg = config or {}

    if hasattr(cfg, "model_dump"):
        cfg = cfg.model_dump()
    elif hasattr(cfg, "dict"):
        cfg = cfg.dict()

    cfg = cfg or {}

    method = str(cfg.get("method", "none")).lower().strip()
    enabled = bool(cfg.get("enabled", method != "none"))

    base_models = cfg.get("base_models") or cfg.get("models") or []
    base_models = [str(model).strip() for model in base_models if str(model).strip()]

    weights = cfg.get("weights")
    if weights is not None:
        try:
            weights = [float(w) for w in weights]
        except (TypeError, ValueError) as exc:
            raise HybridizationError("Hybrid weights must be numeric.") from exc

    return {
        "enabled": enabled,
        "method": method,
        "base_models": base_models,
        "weights": weights,
    }


def validate_hybridization_config(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate and normalize a hybridization configuration."""
    cfg = resolve_hybridization_config(config)

    if not cfg["enabled"]:
        cfg["method"] = "none"
        return cfg

    method = cfg["method"]
    if method not in HYBRID_METHODS or method == "none":
        raise HybridizationError(
            f"Unsupported hybridization method '{method}'. "
            f"Choose from: hard_voting, soft_voting, stacking, blending."
        )

    models = cfg["base_models"]

    if len(models) < 2:
        raise HybridizationError(
            "Hybridization requires at least 2 base ML models."
        )

    invalid = [model for model in models if model not in ML_MODELS]
    if invalid:
        raise HybridizationError(
            "Unknown or non-ML hybrid base model(s): "
            + ", ".join(invalid)
        )

    if len(set(models)) != len(models):
        raise HybridizationError(
            "A hybrid model cannot contain the same base model more than once."
        )

    weights = cfg.get("weights")
    if weights is not None:
        if len(weights) != len(models):
            raise HybridizationError(
                "The number of hybrid weights must match the number "
                "of selected base models."
            )
        if any(weight < 0 for weight in weights):
            raise HybridizationError("Hybrid weights cannot be negative.")
        if sum(weights) <= 0:
            raise HybridizationError(
                "At least one hybrid weight must be greater than zero."
            )

    return cfg


def _build_estimator(
    model_key: str,
    random_state: int,
    class_weight: Optional[dict] = None,
):
    """Build one estimator from the existing ML registry."""
    if model_key not in ML_MODELS:
        raise HybridizationError(
            f"Unknown ML model '{model_key}'."
        )

    spec = ML_MODELS[model_key]

    try:
        return spec.builder(random_state, class_weight)
    except TypeError:
        # Models such as KNN/GaussianNB do not accept class_weight.
        try:
            return spec.builder(random_state, None)
        except Exception as exc:
            raise HybridizationError(
                f"Could not build {spec.label}: {exc}"
            ) from exc
    except Exception as exc:
        raise HybridizationError(
            f"Could not build {spec.label}: {exc}"
        ) from exc


def _build_base_estimators(
    model_keys: Sequence[str],
    random_state: int,
    class_weight: Optional[dict] = None,
) -> List[Tuple[str, Any]]:
    """Build named sklearn estimators for the selected ML models."""
    estimators = []

    for key in model_keys:
        estimator = _build_estimator(
            model_key=key,
            random_state=random_state,
            class_weight=class_weight,
        )
        estimators.append((key, estimator))

    return estimators


def _safe_predict_proba(model, X):
    """
    Return probabilities when available.

    If predict_proba is unavailable, return None rather than silently
    inventing probabilities.
    """
    if not hasattr(model, "predict_proba"):
        return None

    try:
        return np.asarray(model.predict_proba(X))
    except Exception:
        return None


def _build_hybrid_estimator(
    method: str,
    estimators: List[Tuple[str, Any]],
    weights: Optional[List[float]] = None,
):
    """Create the requested sklearn hybrid estimator."""
    if method == "hard_voting":
        return VotingClassifier(
            estimators=estimators,
            voting="hard",
            weights=weights,
            n_jobs=-1,
        )

    if method == "soft_voting":
        # Soft voting requires predict_proba from all base estimators.
        return VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=weights,
            n_jobs=-1,
        )

    if method == "stacking":
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
            stack_method="auto",
            n_jobs=-1,
        )

    raise HybridizationError(
        f"Unsupported sklearn hybrid method '{method}'."
    )


def train_and_evaluate_hybrid(
    config: Dict[str, Any],
    X_train,
    y_train,
    X_test,
    y_test,
    n_classes: int,
    class_weight: Optional[dict] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate one hybrid classifier.

    Blending is implemented explicitly because sklearn does not provide a
    native BlendingClassifier.

    Returns the same broad result structure used by train_ml.py, plus
    hybrid-specific metadata.
    """
    cfg = validate_hybridization_config(config)

    if not cfg["enabled"]:
        raise HybridizationError(
            "Hybridization is disabled."
        )

    method = cfg["method"]
    model_keys = cfg["base_models"]
    weights = cfg.get("weights")

    estimators = _build_base_estimators(
        model_keys=model_keys,
        random_state=random_state,
        class_weight=class_weight,
    )

    start = time.time()

    try:
        if method == "blending":
            fitted_estimators = []

            for key, estimator in estimators:
                estimator.fit(X_train, y_train)
                fitted_estimators.append((key, estimator))

            probabilities = []
            predictions = []

            for key, estimator in fitted_estimators:
                proba = _safe_predict_proba(estimator, X_test)

                if proba is not None:
                    probabilities.append(proba)
                else:
                    predictions.append(
                        np.asarray(estimator.predict(X_test))
                    )

            if not probabilities:
                raise HybridizationError(
                    "Blending requires base models with predict_proba(). "
                    "No selected base model provides probabilities."
                )

            # Use equal or user-specified weights.
            if weights is None:
                normalized_weights = np.ones(len(probabilities))
            else:
                # Weights correspond to the selected models. If a model has
                # no probability output, its prediction cannot participate
                # in probability blending, so use only matching probability
                # models below.
                probability_indices = []
                for index, (_, estimator) in enumerate(fitted_estimators):
                    if _safe_predict_proba(estimator, X_test) is not None:
                        probability_indices.append(index)

                normalized_weights = np.asarray(
                    [weights[i] for i in probability_indices],
                    dtype=float,
                )

                if normalized_weights.sum() <= 0:
                    raise HybridizationError(
                        "Blending probability weights must sum to more than zero."
                    )

            stacked = np.stack(probabilities, axis=0)

            normalized_weights = (
                normalized_weights / normalized_weights.sum()
            )

            weighted_probability = np.average(
                stacked,
                axis=0,
                weights=normalized_weights,
            )

            y_pred = np.argmax(
                weighted_probability,
                axis=1,
            )

            y_proba = weighted_probability

            fitted_model = None

        else:
            fitted_model = _build_hybrid_estimator(
                method=method,
                estimators=estimators,
                weights=weights,
            )

            fitted_model.fit(X_train, y_train)
            y_pred = fitted_model.predict(X_test)
            y_proba = _safe_predict_proba(
                fitted_model,
                X_test,
            )

        training_time = time.time() - start

    except HybridizationError:
        raise
    except Exception as exc:
        raise HybridizationError(
            f"{HYBRID_METHODS.get(method, method)} could not be trained: {exc}"
        ) from exc

    metrics = compute_metrics(
        y_test,
        y_pred,
        y_proba,
        n_classes=n_classes,
    )

    return {
        "model_key": f"hybrid_{method}",
        "model_label": HYBRID_METHODS.get(
            method,
            method.replace("_", " ").title(),
        ),
        "family": "Hybrid",
        "training_time_sec": round(training_time, 3),
        "feature_importance": None,
        "hybridization": {
            "enabled": True,
            "method": method,
            "base_models": model_keys,
            "base_model_labels": [
                ML_MODELS[key].label
                for key in model_keys
            ],
            "weights": weights,
        },
        **metrics,
    }


def get_hybrid_capabilities() -> Dict[str, Any]:
    """Return metadata suitable for exposing hybrid options to the frontend."""
    return {
        "methods": [
            {
                "key": "hard_voting",
                "label": "Hard Voting",
                "description": (
                    "Each base model votes for a class; the majority vote "
                    "becomes the final prediction."
                ),
            },
            {
                "key": "soft_voting",
                "label": "Soft Voting",
                "description": (
                    "Combines class probabilities from the selected base "
                    "models."
                ),
            },
            {
                "key": "stacking",
                "label": "Stacking",
                "description": (
                    "Uses base-model predictions as inputs to a "
                    "meta-classifier."
                ),
            },
            {
                "key": "blending",
                "label": "Blending",
                "description": (
                    "Combines base-model probability predictions using "
                    "equal or user-specified weights."
                ),
            },
        ],
        "eligible_models": [
            {
                "key": key,
                "label": spec.label,
            }
            for key, spec in ML_MODELS.items()
        ],
    }
