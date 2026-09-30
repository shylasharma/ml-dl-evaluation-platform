"""
Dimensionality reduction utilities.

PCA is applied after preprocessing and feature selection.
The PCA transformer is ALWAYS fitted on training data only
to prevent data leakage.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.decomposition import PCA


class PCAError(ValueError):
    """Raised when PCA configuration is invalid."""
    pass


def fit_transform_pca(
    X_train,
    X_test,
    feature_names=None,
    config: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
):
    """
    Fit PCA on X_train only and transform both X_train and X_test.

    Returns:
        X_train_reduced
        X_test_reduced
        result
    """

    cfg = config or {}

    enabled = bool(cfg.get("enabled", False))

    if not enabled:
        original_count = (
            X_train.shape[1]
            if hasattr(X_train, "shape") and len(X_train.shape) > 1
            else len(feature_names or [])
        )

        return (
            X_train,
            X_test,
            {
                "enabled": False,
                "method": "none",
                "original_feature_count": int(original_count),
                "component_count": int(original_count),
                "explained_variance_ratio": [],
                "cumulative_explained_variance": [],
                "variance_retained": 1.0,
                "component_names": feature_names or [],
            },
        )

    if not hasattr(X_train, "shape") or len(X_train.shape) != 2:
        raise PCAError("PCA requires a 2-dimensional feature matrix.")

    original_count = X_train.shape[1]

    if original_count < 2:
        raise PCAError(
            "PCA requires at least 2 features. "
            f"Current feature count: {original_count}."
        )

    mode = cfg.get("mode", "variance")

    # ---------------------------------------------------------
    # Determine PCA configuration
    # ---------------------------------------------------------

    if mode == "components":
        n_components = cfg.get("n_components")

        if n_components is None:
            raise PCAError(
                "Please specify the number of PCA components."
            )

        try:
            n_components = int(n_components)
        except (TypeError, ValueError):
            raise PCAError(
                "PCA component count must be an integer."
            )

        if n_components < 1:
            raise PCAError(
                "PCA component count must be at least 1."
            )

        if n_components > original_count:
            raise PCAError(
                f"PCA components ({n_components}) cannot exceed "
                f"the number of available features ({original_count})."
            )

        pca = PCA(
            n_components=n_components,
            random_state=random_state,
        )

    else:
        # Variance mode
        variance = cfg.get("variance", 0.95)

        try:
            variance = float(variance)
        except (TypeError, ValueError):
            raise PCAError(
                "PCA variance must be a number between 0 and 1."
            )

        if not 0 < variance <= 1:
            raise PCAError(
                "PCA variance must be between 0 and 1."
            )

        pca = PCA(
            n_components=variance,
            random_state=random_state,
        )

    # ---------------------------------------------------------
    # IMPORTANT: FIT ONLY ON TRAINING DATA
    # ---------------------------------------------------------

    X_train_reduced = pca.fit_transform(X_train)

    # Test data is ONLY transformed using the fitted PCA.
    X_test_reduced = pca.transform(X_test)

    explained_variance_ratio = (
        pca.explained_variance_ratio_.tolist()
    )

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    ).tolist()

    component_count = X_train_reduced.shape[1]

    component_names = [
        f"PC{i + 1}"
        for i in range(component_count)
    ]

    return (
        X_train_reduced,
        X_test_reduced,
        {
            "enabled": True,
            "method": "pca",
            "original_feature_count": int(original_count),
            "component_count": int(component_count),
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_explained_variance": cumulative_variance,
            "variance_retained": float(
                cumulative_variance[-1]
            ),
            "component_names": component_names,
        },
    )