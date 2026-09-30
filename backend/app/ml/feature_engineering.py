"""
Feature engineering utilities.

Feature engineering is applied AFTER preprocessing and BEFORE
feature selection / PCA.

Supported transformations:
    - Polynomial features
    - Interaction features
    - Log transformation
    - Ratio features

The transformation is fitted/derived using training data only
where a fitted transformer is required, and the same transformation
is then applied to validation/test data.

Expected pipeline:

    Preprocessing
        ↓
    Feature Engineering
        ↓
    Feature Selection
        ↓
    PCA
        ↓
    Imbalance Handling
        ↓
    Model Training
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_FEATURE_ENGINEERING = {
    "enabled": False,
    "polynomial": False,
    "polynomial_degree": 2,
    "interactions": False,
    "log_transform": False,
    "ratio_features": False,
    "max_interaction_features": None,
}


VALID_POLYNOMIAL_DEGREES = {2, 3}


# ============================================================================
# ERRORS
# ============================================================================


class FeatureEngineeringError(ValueError):
    """Raised when feature-engineering configuration is invalid."""

    pass


# ============================================================================
# CONFIGURATION
# ============================================================================


def resolve_feature_engineering_config(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve and validate feature-engineering configuration.

    Parameters
    ----------
    config:
        Raw configuration dictionary.

    Returns
    -------
    Dict[str, Any]
        Fully resolved configuration.
    """

    cfg = DEFAULT_FEATURE_ENGINEERING.copy()

    if config is None:
        return cfg

    # Support both:
    #
    # {
    #     "enabled": True,
    #     ...
    # }
    #
    # and:
    #
    # {
    #     "feature_engineering": {
    #         "enabled": True,
    #         ...
    #     }
    # }
    if "feature_engineering" in config:
        nested = config.get("feature_engineering")

        if nested is None:
            return cfg

        if not isinstance(nested, dict):
            try:
                nested = nested.model_dump()
            except AttributeError:
                raise FeatureEngineeringError(
                    "Feature engineering configuration must be a dictionary."
                )

        config = nested

    cfg.update(config)

    cfg["enabled"] = bool(cfg.get("enabled", False))

    cfg["polynomial"] = bool(
        cfg.get("polynomial", False)
    )

    cfg["interactions"] = bool(
        cfg.get("interactions", False)
    )

    cfg["log_transform"] = bool(
        cfg.get("log_transform", False)
    )

    cfg["ratio_features"] = bool(
        cfg.get("ratio_features", False)
    )

    try:
        degree = int(
            cfg.get("polynomial_degree", 2)
        )
    except (TypeError, ValueError):
        raise FeatureEngineeringError(
            "Polynomial degree must be an integer."
        )

    if degree not in VALID_POLYNOMIAL_DEGREES:
        raise FeatureEngineeringError(
            "Polynomial degree must be either 2 or 3."
        )

    cfg["polynomial_degree"] = degree

    max_features = cfg.get(
        "max_interaction_features"
    )

    if max_features is not None:

        try:
            max_features = int(max_features)
        except (TypeError, ValueError):
            raise FeatureEngineeringError(
                "max_interaction_features must be an integer."
            )

        if max_features < 1:
            raise FeatureEngineeringError(
                "max_interaction_features must be at least 1."
            )

    cfg["max_interaction_features"] = max_features

    # If FE is disabled, no transformation should be performed.
    if not cfg["enabled"]:
        cfg["polynomial"] = False
        cfg["interactions"] = False
        cfg["log_transform"] = False
        cfg["ratio_features"] = False

    return cfg


# ============================================================================
# HELPERS
# ============================================================================


def _to_dense(X):
    """
    Convert sparse matrices/dataframes into a dense NumPy array.
    """

    if hasattr(X, "toarray"):
        X = X.toarray()

    if hasattr(X, "values"):
        X = X.values

    X = np.asarray(X)

    if X.ndim != 2:
        raise FeatureEngineeringError(
            "Feature engineering requires a 2-dimensional feature matrix."
        )

    return X.astype(float, copy=False)


def _safe_feature_names(
    feature_names: Optional[List[str]],
    n_features: int,
) -> List[str]:
    """
    Ensure feature-name list matches the feature matrix.
    """

    if feature_names is not None:

        names = [
            str(name)
            for name in feature_names
        ]

        if len(names) == n_features:
            return names

    return [
        f"feature_{index + 1}"
        for index in range(n_features)
    ]


def _safe_log_transform(X):
    """
    Apply a signed log transformation.

    Standard log1p cannot handle values below -1.
    Therefore we use:

        sign(x) * log1p(abs(x))

    This works for positive, zero and negative values.
    """

    return np.sign(X) * np.log1p(
        np.abs(X)
    )


def _create_interaction_features(
    X_train,
    X_test,
    feature_names,
    max_features=None,
):
    """
    Create pairwise interaction features.

    To prevent feature explosion, the number of source features
    can be limited with max_features.

    Only unique pairs are generated:

        x1*x2
        x1*x3
        x2*x3
        ...

    Returns
    -------
    train_interactions
    test_interactions
    interaction_names
    """

    n_features = X_train.shape[1]

    if n_features < 2:
        return (
            np.empty(
                (X_train.shape[0], 0)
            ),
            np.empty(
                (X_test.shape[0], 0)
            ),
            [],
        )

    source_count = n_features

    if max_features is not None:
        source_count = min(
            n_features,
            int(max_features),
        )

    train_parts = []
    test_parts = []
    names = []

    for i in range(source_count):

        for j in range(i + 1, source_count):

            train_product = (
                X_train[:, i]
                * X_train[:, j]
            )

            test_product = (
                X_test[:, i]
                * X_test[:, j]
            )

            train_parts.append(
                train_product.reshape(-1, 1)
            )

            test_parts.append(
                test_product.reshape(-1, 1)
            )

            names.append(
                f"{feature_names[i]}_x_{feature_names[j]}"
            )

    if not train_parts:

        return (
            np.empty(
                (X_train.shape[0], 0)
            ),
            np.empty(
                (X_test.shape[0], 0)
            ),
            [],
        )

    return (
        np.hstack(train_parts),
        np.hstack(test_parts),
        names,
    )


def _create_ratio_features(
    X_train,
    X_test,
    feature_names,
    max_features=None,
):
    """
    Create safe pairwise ratio features.

    For every pair:

        feature_i / feature_j

    Division by zero is handled safely by returning zero.

    To control dimensionality, only the first max_features source
    features are used when the limit is supplied.
    """

    n_features = X_train.shape[1]

    if n_features < 2:
        return (
            np.empty(
                (X_train.shape[0], 0)
            ),
            np.empty(
                (X_test.shape[0], 0)
            ),
            [],
        )

    source_count = n_features

    if max_features is not None:
        source_count = min(
            n_features,
            int(max_features),
        )

    train_parts = []
    test_parts = []
    names = []

    for i in range(source_count):

        for j in range(source_count):

            if i == j:
                continue

            denominator_train = X_train[:, j]
            denominator_test = X_test[:, j]

            train_ratio = np.divide(
                X_train[:, i],
                denominator_train,
                out=np.zeros_like(
                    X_train[:, i],
                    dtype=float,
                ),
                where=np.abs(
                    denominator_train
                ) > 1e-12,
            )

            test_ratio = np.divide(
                X_test[:, i],
                denominator_test,
                out=np.zeros_like(
                    X_test[:, i],
                    dtype=float,
                ),
                where=np.abs(
                    denominator_test
                ) > 1e-12,
            )

            train_parts.append(
                train_ratio.reshape(-1, 1)
            )

            test_parts.append(
                test_ratio.reshape(-1, 1)
            )

            names.append(
                f"{feature_names[i]}_div_{feature_names[j]}"
            )

    if not train_parts:

        return (
            np.empty(
                (X_train.shape[0], 0)
            ),
            np.empty(
                (X_test.shape[0], 0)
            ),
            [],
        )

    return (
        np.hstack(train_parts),
        np.hstack(test_parts),
        names,
    )


# ============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# ============================================================================


def fit_transform_feature_engineering(
    X_train,
    X_test,
    feature_names: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply feature engineering to training and test/validation data.

    Parameters
    ----------
    X_train:
        Training feature matrix.

    X_test:
        Validation/test feature matrix.

    feature_names:
        Names of the input features.

    config:
        Feature-engineering configuration.

    random_state:
        Reserved for reproducibility and future stochastic
        feature-engineering methods.

    Returns
    -------
    X_train_engineered
    X_test_engineered
    result
    """

    del random_state  # Reserved for future stochastic methods.

    cfg = resolve_feature_engineering_config(
        config
    )

    X_train = _to_dense(X_train)
    X_test = _to_dense(X_test)

    if X_train.shape[1] != X_test.shape[1]:
        raise FeatureEngineeringError(
            "Training and validation/test matrices must have "
            "the same number of features."
        )

    original_feature_count = X_train.shape[1]

    names = _safe_feature_names(
        feature_names,
        original_feature_count,
    )

    # ------------------------------------------------------------------------
    # FEATURE ENGINEERING DISABLED
    # ------------------------------------------------------------------------

    if not cfg["enabled"]:

        return (
            X_train,
            X_test,
            {
                "enabled": False,
                "methods": [],
                "original_feature_count": int(
                    original_feature_count
                ),
                "engineered_feature_count": int(
                    original_feature_count
                ),
                "new_feature_count": 0,
                "feature_names": names,
                "new_feature_names": [],
            },
        )

    # ------------------------------------------------------------------------
    # WORKING MATRICES
    # ------------------------------------------------------------------------

    train_blocks = [X_train]
    test_blocks = [X_test]

    engineered_names = []
    methods_used = []

    # ------------------------------------------------------------------------
    # LOG TRANSFORMATION
    # ------------------------------------------------------------------------

    if cfg["log_transform"]:

        X_train_log = _safe_log_transform(
            X_train
        )

        X_test_log = _safe_log_transform(
            X_test
        )

        train_blocks.append(
            X_train_log
        )

        test_blocks.append(
            X_test_log
        )

        engineered_names.extend(
            [
                f"log_{name}"
                for name in names
            ]
        )

        methods_used.append(
            "log_transform"
        )

    # ------------------------------------------------------------------------
    # POLYNOMIAL FEATURES
    # ------------------------------------------------------------------------

    if cfg["polynomial"]:

        polynomial = PolynomialFeatures(
            degree=cfg["polynomial_degree"],
            include_bias=False,
        )

        X_train_poly = polynomial.fit_transform(
            X_train
        )

        X_test_poly = polynomial.transform(
            X_test
        )

        polynomial_names = list(
            polynomial.get_feature_names_out(
                names
            )
        )

        # Keep only genuinely new polynomial features.
        original_name_count = len(names)

        if (
            X_train_poly.shape[1]
            > original_name_count
        ):

            X_train_poly = X_train_poly[
                :,
                original_name_count:,
            ]

            X_test_poly = X_test_poly[
                :,
                original_name_count:,
            ]

            polynomial_names = polynomial_names[
                original_name_count:
            ]

            train_blocks.append(
                X_train_poly
            )

            test_blocks.append(
                X_test_poly
            )

            engineered_names.extend(
                polynomial_names
            )

        methods_used.append(
            "polynomial"
        )

    # ------------------------------------------------------------------------
    # INTERACTION FEATURES
    # ------------------------------------------------------------------------

    if cfg["interactions"]:

        (
            X_train_interactions,
            X_test_interactions,
            interaction_names,
        ) = _create_interaction_features(
            X_train=X_train,
            X_test=X_test,
            feature_names=names,
            max_features=cfg[
                "max_interaction_features"
            ],
        )

        if X_train_interactions.shape[1] > 0:

            train_blocks.append(
                X_train_interactions
            )

            test_blocks.append(
                X_test_interactions
            )

            engineered_names.extend(
                interaction_names
            )

        methods_used.append(
            "interactions"
        )

    # ------------------------------------------------------------------------
    # RATIO FEATURES
    # ------------------------------------------------------------------------

    if cfg["ratio_features"]:

        (
            X_train_ratios,
            X_test_ratios,
            ratio_names,
        ) = _create_ratio_features(
            X_train=X_train,
            X_test=X_test,
            feature_names=names,
            max_features=cfg[
                "max_interaction_features"
            ],
        )

        if X_train_ratios.shape[1] > 0:

            train_blocks.append(
                X_train_ratios
            )

            test_blocks.append(
                X_test_ratios
            )

            engineered_names.extend(
                ratio_names
            )

        methods_used.append(
            "ratio_features"
        )

    # ------------------------------------------------------------------------
    # FINAL MATRICES
    # ------------------------------------------------------------------------

    X_train_engineered = np.hstack(
        train_blocks
    )

    X_test_engineered = np.hstack(
        test_blocks
    )

    # Guard against NaN / infinity generated by unusual data.
    X_train_engineered = np.nan_to_num(
        X_train_engineered,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_test_engineered = np.nan_to_num(
        X_test_engineered,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    final_feature_names = (
        names + engineered_names
    )

    return (
        X_train_engineered,
        X_test_engineered,
        {
            "enabled": True,
            "methods": methods_used,
            "original_feature_count": int(
                original_feature_count
            ),
            "engineered_feature_count": int(
                X_train_engineered.shape[1]
            ),
            "new_feature_count": int(
                X_train_engineered.shape[1]
                - original_feature_count
            ),
            "feature_names": final_feature_names,
            "new_feature_names": engineered_names,
            "polynomial_degree": (
                cfg["polynomial_degree"]
                if cfg["polynomial"]
                else None
            ),
            "max_interaction_features": cfg[
                "max_interaction_features"
            ],
        },
    )


# ============================================================================
# BACKWARD-COMPATIBILITY ALIAS
# ============================================================================


apply_feature_engineering = (
    fit_transform_feature_engineering
)