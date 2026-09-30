"""
Feature Selection utilities.

Research-methodology rules enforced here:

1. Feature selection is FIT ONLY on training data.
2. The fitted selector is then used to TRANSFORM both training and
   test/validation data.
3. Test/validation data is never used to calculate feature-selection
   scores or determine the selected features.
4. Feature selection is performed on the actual transformed feature
   matrix produced by the preprocessing pipeline.
5. Original/transformed feature names are preserved for research
   reporting.
6. The module supports filter, wrapper and embedded feature-selection
   strategies.

Supported methods
-----------------
FILTER:
    - correlation
    - chi2
    - anova
    - mutual_information

WRAPPER:
    - rfe
    - sequential

EMBEDDED:
    - l1
    - tree_importance

The module is intentionally independent from preprocessing, PCA and
model training so that each stage can be tested and reused separately.
"""

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    RFE,
    SequentialFeatureSelector,
    SelectFromModel,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mutual_info_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALID_METHODS = {
    "none",
    "correlation",
    "chi2",
    "anova",
    "mutual_information",
    "rfe",
    "sequential",
    "l1",
    "tree_importance",
}


FILTER_METHODS = {
    "correlation",
    "chi2",
    "anova",
    "mutual_information",
}


WRAPPER_METHODS = {
    "rfe",
    "sequential",
}


EMBEDDED_METHODS = {
    "l1",
    "tree_importance",
}


DEFAULT_FEATURE_SELECTION: Dict[str, Any] = {
    "enabled": False,
    "method": "none",
    "k": None,
    "percentage": None,
    "threshold": None,
    "correlation_threshold": 0.95,
}


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def resolve_feature_selection_config(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve and validate feature-selection configuration.

    Backward compatibility:
        If no feature-selection configuration is supplied, feature selection
        remains disabled.

    Supported configuration:

        {
            "enabled": True,
            "method": "mutual_information",
            "k": 20
        }

    OR:

        {
            "enabled": True,
            "method": "mutual_information",
            "percentage": 50
        }

    OR:

        {
            "enabled": True,
            "method": "l1",
            "threshold": "median"
        }
    """

    if not config:
        return dict(DEFAULT_FEATURE_SELECTION)

    resolved = {
        **DEFAULT_FEATURE_SELECTION,
        **{
            key: value
            for key, value in dict(config).items()
            if value is not None
        },
    }

    enabled = bool(resolved["enabled"])
    method = str(resolved["method"]).lower().strip()

    if not enabled:
        resolved["enabled"] = False
        resolved["method"] = "none"
        return resolved

    if method not in VALID_METHODS:
        raise ValueError(
            f"Unsupported feature-selection method '{method}'. "
            f"Supported methods: {', '.join(sorted(VALID_METHODS))}."
        )

    if method == "none":
        resolved["enabled"] = False
        return resolved

    resolved["enabled"] = True
    resolved["method"] = method

    # k validation
    if resolved.get("k") is not None:
        try:
            k = int(resolved["k"])
        except (TypeError, ValueError):
            raise ValueError("Feature-selection 'k' must be an integer.")

        if k < 1:
            raise ValueError("Feature-selection 'k' must be >= 1.")

        resolved["k"] = k

    # percentage validation
    if resolved.get("percentage") is not None:
        try:
            percentage = float(resolved["percentage"])
        except (TypeError, ValueError):
            raise ValueError(
                "Feature-selection 'percentage' must be numeric."
            )

        if not 0 < percentage <= 100:
            raise ValueError(
                "Feature-selection 'percentage' must be between 0 and 100."
            )

        resolved["percentage"] = percentage

    # threshold is mainly used by embedded methods.
    if resolved.get("threshold") is not None:
        threshold = resolved["threshold"]

        if isinstance(threshold, str):
            allowed = {
                "mean",
                "median",
                "mean",
                "1.25*mean",
                "0.5*mean",
            }

            if threshold not in allowed:
                try:
                    threshold = float(threshold)
                except ValueError:
                    raise ValueError(
                        "Unsupported feature-selection threshold."
                    )

        elif not isinstance(threshold, (int, float)):
            raise ValueError(
                "Feature-selection 'threshold' must be numeric or a "
                "supported string such as 'mean' or 'median'."
            )

        resolved["threshold"] = threshold

    # Correlation-specific configuration.
    if method == "correlation":
        threshold = resolved.get("correlation_threshold", 0.95)

        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            raise ValueError(
                "Correlation threshold must be numeric."
            )

        if not 0 < threshold <= 1:
            raise ValueError(
                "Correlation threshold must be between 0 and 1."
            )

        resolved["correlation_threshold"] = threshold

    return resolved


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _ensure_2d_array(X) -> np.ndarray:
    """
    Convert a feature matrix into a dense 2D numpy array.

    OneHotEncoder may produce sparse matrices depending on the sklearn
    version/configuration. The current platform already converts these
    matrices to dense arrays, but this function keeps the module robust.
    """

    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            f"Feature-selection input must be 2-dimensional; "
            f"received shape {X.shape}."
        )

    return X


def _clean_feature_names(
    feature_names: Optional[List[str]],
    n_features: int,
) -> List[str]:
    """
    Ensure that there is exactly one readable name per transformed feature.
    """

    if feature_names is None:
        return [f"feature_{i}" for i in range(n_features)]

    names = list(feature_names)

    if len(names) != n_features:
        return [
            f"feature_{i}"
            for i in range(n_features)
        ]

    return [str(name) for name in names]


def _resolve_k(
    n_features: int,
    k: Optional[int] = None,
    percentage: Optional[float] = None,
) -> int:
    """
    Convert k/percentage configuration into a valid number of features.
    """

    if n_features < 1:
        raise ValueError("No features are available for feature selection.")

    if k is not None:
        selected_k = int(k)

    elif percentage is not None:
        selected_k = int(np.ceil(
            n_features * float(percentage) / 100.0
        ))

    else:
        # If nothing is supplied, retain approximately half the features.
        selected_k = max(1, n_features // 2)

    selected_k = max(1, min(selected_k, n_features))

    return selected_k


def _safe_numeric_matrix(X) -> np.ndarray:
    """
    Convert input to a numeric matrix.

    Feature selection happens AFTER the preprocessing pipeline, so the
    expected input should already be numerical.
    """

    X = _ensure_2d_array(X)

    if not np.issubdtype(X.dtype, np.number):
        try:
            X = X.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Feature selection requires a numerical feature matrix. "
                "Run preprocessing/encoding before feature selection."
            ) from exc

    # Replace non-finite values defensively.
    if not np.isfinite(X).all():
        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    return X.astype(float)


# ---------------------------------------------------------------------------
# Correlation filter
# ---------------------------------------------------------------------------

def _correlation_selection(
    X_train: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove highly correlated redundant features.

    The correlation matrix is calculated ONLY from X_train.

    For every highly correlated pair, the later feature is removed.

    Returns:
        selected_indices
        scores
    """

    X = _safe_numeric_matrix(X_train)

    if X.shape[1] <= 1:
        return np.arange(X.shape[1]), np.ones(X.shape[1])

    corr = np.corrcoef(X, rowvar=False)

    # Numerical protection for constant columns.
    corr = np.nan_to_num(corr, nan=0.0)

    upper = np.triu(np.abs(corr), k=1)

    to_remove = set()

    for i in range(upper.shape[0]):
        for j in range(i + 1, upper.shape[1]):
            if upper[i, j] >= threshold:
                to_remove.add(j)

    selected = [
        i for i in range(X.shape[1])
        if i not in to_remove
    ]

    if not selected:
        # Safety fallback: always retain at least one feature.
        selected = [0]

    # Correlation is a redundancy measure rather than a target-relevance
    # score. We report the maximum absolute correlation with another feature
    # as an explanatory score.
    scores = np.max(np.abs(corr), axis=1)

    return np.asarray(selected, dtype=int), scores


# ---------------------------------------------------------------------------
# Generic SelectKBest methods
# ---------------------------------------------------------------------------

def _select_k_best(
    X_train: np.ndarray,
    y_train: np.ndarray,
    score_function,
    k: int,
):
    """
    Fit SelectKBest on training data only.
    """

    selector = SelectKBest(score_func=score_function, k=k)
    selector.fit(X_train, y_train)

    return selector


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_feature_selector(
    X_train,
    y_train,
    feature_names: Optional[List[str]] = None,
    method: str = "none",
    k: Optional[int] = None,
    percentage: Optional[float] = None,
    threshold: Optional[Any] = None,
    correlation_threshold: float = 0.95,
    random_state: int = 42,
):
    """
    Fit a feature selector using TRAINING DATA ONLY.

    Parameters
    ----------
    X_train:
        Preprocessed training feature matrix.

    y_train:
        Training labels.

    feature_names:
        Names of the transformed features.

    method:
        Feature-selection strategy.

    k:
        Number of features to retain.

    percentage:
        Percentage of features to retain.

    threshold:
        Embedded-method threshold.

    correlation_threshold:
        Threshold used by correlation filtering.

    random_state:
        Reproducibility seed.

    Returns
    -------
    selector_result : dict

        Contains the fitted selector plus research/reporting metadata.

    IMPORTANT:
        This function does not touch test data.
    """

    X_train = _safe_numeric_matrix(X_train)

    y_train = np.asarray(y_train)

    n_features = X_train.shape[1]

    feature_names = _clean_feature_names(
        feature_names,
        n_features,
    )

    method = str(method or "none").lower().strip()

    if method not in VALID_METHODS:
        raise ValueError(
            f"Unsupported feature-selection method '{method}'."
        )

    # ------------------------------------------------------------------
    # No feature selection
    # ------------------------------------------------------------------

    if method == "none":
        return {
            "selector": None,
            "method": "none",
            "method_category": "none",
            "enabled": False,
            "original_feature_count": n_features,
            "selected_feature_count": n_features,
            "original_feature_names": feature_names,
            "selected_feature_names": feature_names,
            "selected_indices": list(range(n_features)),
            "scores": {},
            "summary": (
                f"{n_features} features retained "
                "(feature selection disabled)."
            ),
        }

    # ------------------------------------------------------------------
    # FILTER — correlation
    # ------------------------------------------------------------------

    if method == "correlation":
        selected_indices, scores = _correlation_selection(
            X_train,
            feature_names,
            threshold=correlation_threshold,
        )

        return {
            "selector": None,
            "method": method,
            "method_category": "filter",
            "enabled": True,
            "original_feature_count": n_features,
            "selected_feature_count": len(selected_indices),
            "original_feature_names": feature_names,
            "selected_feature_names": [
                feature_names[i] for i in selected_indices
            ],
            "selected_indices": selected_indices.tolist(),
            "scores": {
                feature_names[i]: float(scores[i])
                for i in range(len(feature_names))
            },
            "threshold": correlation_threshold,
            "summary": (
                f"{n_features} → {len(selected_indices)} features "
                f"using correlation filtering."
            ),
        }

    # ------------------------------------------------------------------
    # Resolve k for SelectKBest / wrappers
    # ------------------------------------------------------------------

    selected_k = _resolve_k(
        n_features,
        k=k,
        percentage=percentage,
    )

    # ------------------------------------------------------------------
    # FILTER — Chi-square
    # ------------------------------------------------------------------

    if method == "chi2":
        # chi2 requires non-negative values.
        # MinMax scaling is applied ONLY to the training matrix used for
        # calculating the selector scores.
        selector_scaler = MinMaxScaler()
        X_non_negative = selector_scaler.fit_transform(X_train)

        selector = _select_k_best(
            X_non_negative,
            y_train,
            chi2,
            selected_k,
        )

        scores = selector.scores_
        selected_indices = np.where(selector.get_support())[0]

        return {
            "selector": {
                "type": "chi2",
                "sklearn_selector": selector,
                "scaler": selector_scaler,
            },
            "method": method,
            "method_category": "filter",
            "enabled": True,
            "original_feature_count": n_features,
            "selected_feature_count": len(selected_indices),
            "original_feature_names": feature_names,
            "selected_feature_names": [
                feature_names[i] for i in selected_indices
            ],
            "selected_indices": selected_indices.tolist(),
            "scores": {
                feature_names[i]: (
                    float(scores[i])
                    if np.isfinite(scores[i])
                    else 0.0
                )
                for i in range(len(feature_names))
            },
            "k": selected_k,
            "summary": (
                f"{n_features} → {len(selected_indices)} features "
                f"using Chi-square selection."
            ),
        }

    # ------------------------------------------------------------------
    # FILTER — ANOVA F-test
    # ------------------------------------------------------------------

    if method == "anova":
        selector = _select_k_best(
            X_train,
            y_train,
            f_classif,
            selected_k,
        )

        scores = selector.scores_
        selected_indices = np.where(selector.get_support())[0]

        return {
            "selector": selector,
            "method": method,
            "method_category": "filter",
            "enabled": True,
            "original_feature_count": n_features,
            "selected_feature_count": len(selected_indices),
            "original_feature_names": feature_names,
            "selected_feature_names": [
                feature_names[i] for i in selected_indices
            ],
            "selected_indices": selected_indices.tolist(),
            "scores": {
                feature_names[i]: (
                    float(scores[i])
                    if np.isfinite(scores[i])
                    else 0.0
                )
                for i in range(len(feature_names))
            },
            "k": selected_k,
            "summary": (
                f"{n_features} → {len(selected_indices)} features "
                f"using ANOVA F-test."
            ),
        }

    # ------------------------------------------------------------------
    # FILTER — Mutual Information
    # ------------------------------------------------------------------

    if method == "mutual_information":
        selector = _select_k_best(
            X_train,
            y_train,
            mutual_info_classif,
            selected_k,
        )

        scores = selector.scores_
        selected_indices = np.where(selector.get_support())[0]

        return {
            "selector": selector,
            "method": method,
            "method_category": "filter",
            "enabled": True,
            "original_feature_count": n_features,
            "selected_feature_count": len(selected_indices),
            "original_feature_names": feature_names,
            "selected_feature_names": [
                feature_names[i] for i in selected_indices
            ],
            "selected_indices": selected_indices.tolist(),
            "scores": {
                feature_names[i]: (
                    float(scores[i])
                    if np.isfinite(scores[i])
                    else 0.0
                )
                for i in range(len(feature_names))
            },
            "k": selected_k,
            "summary": (
                f"{n_features} → {len(selected_indices)} features "
                f"using Mutual Information."
            ),
        }

    # ------------------------------------------------------------------
    # WRAPPER — RFE
    # ------------------------------------------------------------------

    if method == "rfe":
        estimator = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=random_state,
        )

        selector = RFE(
            estimator=estimator,
            n_features_to_select=selected_k,
            step=1,
        )

        selector.fit(X_train, y_train)

        selected_indices = np.where(selector.support_)[0]

        ranking = selector.ranking_

        return {
            "selector": selector,
            "method": method,
            "method_category": "wrapper",
            "enabled": True,
            "original_feature_count": n_features,
            "selected_feature_count": len(selected_indices),
            "original_feature_names": feature_names,
            "selected_feature_names": [
                feature_names[i] for i in selected_indices
            ],
            "selected_indices": selected_indices.tolist(),
            "scores": {
                feature_names[i]: float(
                    1.0 / max(1, ranking[i])
                )
                for i in range(len(feature_names))
            },
            "ranking": {
                feature_names[i]: int(ranking[i])
                for i in range(len(feature_names))
            },
            "k": selected_k,
            "summary": (
                f"{n_features} → {len(selected_indices)} features "
                f"using Recursive Feature Elimination."
            ),
        }

    # ------------------------------------------------------------------
    # WRAPPER — Sequential Feature Selection
    # ------------------------------------------------------------------

    if method == "sequential":
        estimator = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=random_state,
        )

        selector = SequentialFeatureSelector(
            estimator,
            n_features_to_select=selected_k,
            direction="forward",
            scoring="f1_weighted",
            cv=3,
            n_jobs=-1,
        )

        selector.fit(X_train, y_train)

        selected_indices = np.where(selector.get_support())[0]

        return {
            "selector": selector,
            "method": method,
            "method_category": "wrapper",
            "enabled": True,
            "original_feature_count": n_features,
            "selected_feature_count": len(selected_indices),
            "original_feature_names": feature_names,
            "selected_feature_names": [
                feature_names[i] for i in selected_indices
            ],
            "selected_indices": selected_indices.tolist(),
            "scores": {
                feature_names[i]: (
                    1.0 if selector.get_support()[i] else 0.0
                )
                for i in range(len(feature_names))
            },
            "k": selected_k,
            "summary": (
                f"{n_features} → {len(selected_indices)} features "
                f"using Sequential Feature Selection."
            ),
        }

    # ------------------------------------------------------------------
    # EMBEDDED — L1
    # ------------------------------------------------------------------

    if method == "l1":
        estimator = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=2000,
            random_state=random_state,
        )

        if threshold is None:
            threshold = "median"

        selector = SelectFromModel(
            estimator=estimator,
            threshold=threshold,
            max_features=selected_k if k is not None else None,
        )

        selector.fit(X_train, y_train)

        selected_indices = np.where(selector.get_support())[0]

        # Safety fallback.
        if len(selected_indices) == 0:
            selector = SelectFromModel(
                estimator=estimator,
                threshold=-np.inf,
                max_features=selected_k,
            )
            selector.fit(X_train, y_train)
            selected_indices = np.where(selector.get_support())[0]

        importance = np.abs(
            selector.estimator_.coef_
        ).mean(axis=0)

        return {
            "selector": selector,
            "method": method,
            "method_category": "embedded",
            "enabled": True,
            "original_feature_count": n_features,
            "selected_feature_count": len(selected_indices),
            "original_feature_names": feature_names,
            "selected_feature_names": [
                feature_names[i] for i in selected_indices
            ],
            "selected_indices": selected_indices.tolist(),
            "scores": {
                feature_names[i]: float(importance[i])
                for i in range(len(feature_names))
            },
            "threshold": threshold,
            "summary": (
                f"{n_features} → {len(selected_indices)} features "
                f"using L1 embedded selection."
            ),
        }

    # ------------------------------------------------------------------
    # EMBEDDED — Tree importance
    # ------------------------------------------------------------------

    if method == "tree_importance":
        estimator = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

        if threshold is None:
            threshold = "median"

        selector = SelectFromModel(
            estimator=estimator,
            threshold=threshold,
            max_features=selected_k if k is not None else None,
        )

        selector.fit(X_train, y_train)

        selected_indices = np.where(selector.get_support())[0]

        if len(selected_indices) == 0:
            selector = SelectFromModel(
                estimator=estimator,
                threshold=-np.inf,
                max_features=selected_k,
            )
            selector.fit(X_train, y_train)
            selected_indices = np.where(selector.get_support())[0]

        importance = selector.estimator_.feature_importances_

        return {
            "selector": selector,
            "method": method,
            "method_category": "embedded",
            "enabled": True,
            "original_feature_count": n_features,
            "selected_feature_count": len(selected_indices),
            "original_feature_names": feature_names,
            "selected_feature_names": [
                feature_names[i] for i in selected_indices
            ],
            "selected_indices": selected_indices.tolist(),
            "scores": {
                feature_names[i]: float(importance[i])
                for i in range(len(feature_names))
            },
            "threshold": threshold,
            "summary": (
                f"{n_features} → {len(selected_indices)} features "
                f"using tree-based feature importance."
            ),
        }

    raise ValueError(
        f"Feature-selection method '{method}' was not implemented."
    )


# ---------------------------------------------------------------------------
# Transform data using fitted selector
# ---------------------------------------------------------------------------

def transform_with_feature_selector(
    X,
    selector_result: Dict[str, Any],
):
    """
    Transform data using a selector previously fitted on TRAINING data.

    This function can be called for both:

        X_train
        X_test / X_validation

    using the SAME fitted selector.

    No fitting occurs here.
    """

    X = _safe_numeric_matrix(X)

    method = selector_result.get("method", "none")

    if method == "none":
        return X

    selected_indices = selector_result.get("selected_indices")

    if selected_indices is None:
        raise ValueError(
            "Feature-selection result does not contain selected indices."
        )

    # Chi-square has an internal scaler because chi2 requires non-negative
    # values. Apply the exact same fitted scaler used during training.
    if method == "chi2":
        selector_info = selector_result.get("selector")

        if not isinstance(selector_info, dict):
            raise ValueError(
                "Invalid Chi-square selector state."
            )

        scaler = selector_info.get("scaler")
        sklearn_selector = selector_info.get("sklearn_selector")

        if scaler is None or sklearn_selector is None:
            raise ValueError(
                "Chi-square selector is missing its fitted components."
            )

        X_scaled = scaler.transform(X)
        return sklearn_selector.transform(X_scaled)

    selector = selector_result.get("selector")

    # Correlation filtering does not use a sklearn selector.
    if method == "correlation":
        return X[:, selected_indices]

    if selector is None:
        return X[:, selected_indices]

    return selector.transform(X)


# ---------------------------------------------------------------------------
# Complete fit + transform helper
# ---------------------------------------------------------------------------

def fit_transform_feature_selection(
    X_train,
    X_test,
    y_train,
    feature_names: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
):
    """
    Convenience function for the orchestrator.

    Workflow:

        X_train
           ↓
        FIT selector
           ↓
        Transform X_train
           ↓
        Transform X_test using SAME selector

    X_test is NEVER used during fitting.
    """

    resolved = resolve_feature_selection_config(config)

    if not resolved["enabled"]:
        X_train = _safe_numeric_matrix(X_train)
        X_test = _safe_numeric_matrix(X_test)

        names = _clean_feature_names(
            feature_names,
            X_train.shape[1],
        )

        result = {
            "selector": None,
            "method": "none",
            "method_category": "none",
            "enabled": False,
            "original_feature_count": len(names),
            "selected_feature_count": len(names),
            "original_feature_names": names,
            "selected_feature_names": names,
            "selected_indices": list(range(len(names))),
            "scores": {},
            "summary": (
                f"{len(names)} features retained "
                "(feature selection disabled)."
            ),
        }

        return X_train, X_test, result

    result = fit_feature_selector(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        method=resolved["method"],
        k=resolved.get("k"),
        percentage=resolved.get("percentage"),
        threshold=resolved.get("threshold"),
        correlation_threshold=resolved.get(
            "correlation_threshold",
            0.95,
        ),
        random_state=random_state,
    )

    X_train_selected = transform_with_feature_selector(
        X_train,
        result,
    )

    X_test_selected = transform_with_feature_selector(
        X_test,
        result,
    )

    return X_train_selected, X_test_selected, result


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def get_feature_selection_report(
    selector_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return a JSON-safe research/report representation.

    Fitted sklearn objects are deliberately excluded because they are not
    suitable for JSON serialization.
    """

    if not selector_result:
        return {
            "enabled": False,
            "method": "none",
        }

    report = {
        "enabled": bool(selector_result.get("enabled", False)),
        "method": selector_result.get("method", "none"),
        "method_category": selector_result.get(
            "method_category",
            "none",
        ),
        "original_feature_count": int(
            selector_result.get(
                "original_feature_count",
                0,
            )
        ),
        "selected_feature_count": int(
            selector_result.get(
                "selected_feature_count",
                0,
            )
        ),
        "original_feature_names": list(
            selector_result.get(
                "original_feature_names",
                [],
            )
        ),
        "selected_feature_names": list(
            selector_result.get(
                "selected_feature_names",
                [],
            )
        ),
        "selected_indices": [
            int(i)
            for i in selector_result.get(
                "selected_indices",
                [],
            )
        ],
        "scores": {
            str(name): float(value)
            for name, value in selector_result.get(
                "scores",
                {},
            ).items()
            if value is not None and np.isfinite(float(value))
        },
        "summary": selector_result.get("summary", ""),
    }

    if "threshold" in selector_result:
        threshold = selector_result["threshold"]

        if isinstance(threshold, (int, float)):
            report["threshold"] = float(threshold)
        else:
            report["threshold"] = threshold

    if "k" in selector_result:
        report["k"] = (
            int(selector_result["k"])
            if selector_result["k"] is not None
            else None
        )

    if "ranking" in selector_result:
        report["ranking"] = {
            str(name): int(rank)
            for name, rank in selector_result["ranking"].items()
        }

    return report