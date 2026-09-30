"""
Preprocessing utilities.

Critical research-methodology rule enforced here: the train/test split
happens BEFORE any resampling technique (e.g. SMOTE) is applied, and
scaling/encoding is *fit* only on the training fold and re-applied
(transform-only) to the test fold. This avoids data leakage.

Phase 1 (configurable preprocessing) adds explicit, validated choices for
missing-value handling, scaling, encoding and duplicate-row handling, while
preserving the platform's original hard-coded behaviour as the default so
that existing experiment requests keep producing identical results. See
`resolve_preprocessing_config` for exactly how backward compatibility is
guaranteed, and `drop_duplicate_rows` for why duplicate removal is safe to
do before the split even though it operates on the whole dataset.

This module is kept intentionally narrow (impute/scale/encode + dedup) so
that later phases (feature engineering, feature selection, PCA) can each get
their own `resolve_*_config` + apply function here or in sibling modules,
orchestrated the same way from `app/ml/orchestrator.py`, without needing to
restructure this file again.
"""
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from app.utils.errors import FriendlyError

# ---------------------------------------------------------------------------
# Configurable preprocessing (Phase 1)
# ---------------------------------------------------------------------------

VALID_MISSING_NUMERIC = {"median", "mean"}
VALID_MISSING_CATEGORICAL = {"most_frequent"}
VALID_SCALING = {"none", "standard", "minmax", "robust"}
VALID_ENCODING = {"onehot"}
VALID_DUPLICATES = {"keep", "remove"}

# These values are exactly the platform's original hard-coded behaviour.
DEFAULT_PREPROCESSING: Dict[str, str] = {
    "missing_numeric": "median",
    "missing_categorical": "most_frequent",
    "scaling": "standard",
    "encoding": "onehot",
    "duplicates": "keep",
}

_SCALERS = {"none": None, "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def resolve_preprocessing_config(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Turns the raw experiment config dict into a fully-specified, validated
    preprocessing configuration.

    Backward compatibility: if the request has no explicit `preprocessing`
    block (older clients, or any request built before this feature existed),
    an equivalent configuration is derived from the legacy `scale_features`
    boolean, so old experiments produce identical results to before this
    feature was added.

    This is the single place preprocessing choices are validated; callers
    further down the pipeline (`prepare_data`, cross-validation) treat the
    returned dict as already-trusted.
    """
    pp = config.get("preprocessing")
    if not pp:
        legacy_scale = config.get("scale_features", True)
        resolved = dict(DEFAULT_PREPROCESSING)
        resolved["scaling"] = "standard" if legacy_scale else "none"
        return resolved

    resolved = {**DEFAULT_PREPROCESSING, **{k: v for k, v in dict(pp).items() if v is not None}}

    for key, valid_values in (
        ("missing_numeric", VALID_MISSING_NUMERIC),
        ("missing_categorical", VALID_MISSING_CATEGORICAL),
        ("scaling", VALID_SCALING),
        ("encoding", VALID_ENCODING),
        ("duplicates", VALID_DUPLICATES),
    ):
        if resolved[key] not in valid_values:
            raise FriendlyError(
                f"Unsupported preprocessing option '{resolved[key]}' for '{key}'. "
                f"Supported options: {', '.join(sorted(valid_values))}."
            )
    return resolved


def drop_duplicate_rows(df: pd.DataFrame, duplicates_option: str) -> Tuple[pd.DataFrame, int]:
    """
    Optionally removes exact duplicate rows (across every column, including
    the target) from the WHOLE dataset.

    Where: right after dataset profiling, before the train/test split.
    Why here, and why this is leakage-safe: this is a deterministic
    data-cleaning step, not a statistic fit from the data (unlike scaling,
    imputation or resampling), so it does not need to be restricted to
    training rows only. Doing it before the split is actually the safer
    order: if skipped, an exact duplicate pair could otherwise be split
    across train and test, letting a model be evaluated on a row identical to
    one it trained on -- a subtle form of leakage. Removing duplicates up
    front prevents exactly that.
    Fits/learns parameters: no.
    """
    if duplicates_option != "remove":
        return df, 0
    before = len(df)
    deduped = df.drop_duplicates().reset_index(drop=True)
    return deduped, before - len(deduped)


def build_feature_pipeline(numeric_features: List[str], categorical_features: List[str],
                            preprocessing: Optional[Dict[str, str]] = None) -> ColumnTransformer:
    """
    Builds the per-column impute+scale (numeric) / impute+encode (categorical)
    pipeline. `preprocessing` must be an already-resolved, validated config
    (see `resolve_preprocessing_config`); omitting it reproduces the
    platform's original defaults (median / most_frequent / StandardScaler /
    one-hot).
    """
    cfg = preprocessing or DEFAULT_PREPROCESSING

    numeric_steps = [("imputer", SimpleImputer(strategy=cfg.get("missing_numeric", "median")))]
    scaler_cls = _SCALERS.get(cfg.get("scaling", "standard"), StandardScaler)
    if scaler_cls is not None:
        numeric_steps.append(("scaler", scaler_cls()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg.get("missing_categorical", "most_frequent"))),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers, remainder="drop")


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    numeric_features: List[str],
    categorical_features: List[str],
    test_size: float = 0.25,
    random_state: int = 42,
    scale_features: bool = True,
    preprocessing: Optional[Dict[str, str]] = None,
):
    """
    Returns X_train, X_test (dense numpy arrays), y_train, y_test (encoded
    integer labels), the fitted feature pipeline, the fitted label encoder,
    the number of classes, and the resolved preprocessing config actually
    used (so callers can report exactly what was applied).

    `preprocessing`, when given, must already be resolved and validated (see
    `resolve_preprocessing_config`). When omitted, the legacy `scale_features`
    boolean is used instead, so any existing direct caller of this function
    keeps working exactly as before Phase 1.

    Fitting rule (unchanged): the feature pipeline is fit on the training
    split only and merely applied (transform-only) to the test split.
    """
    resolved = preprocessing or {**DEFAULT_PREPROCESSING, "scaling": "standard" if scale_features else "none"}

    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    if df.empty:
        raise FriendlyError(
            f"The target column '{target_column}' has no values, so there is nothing to predict. "
            f"Check that your CSV's label/target column is filled in (a trailing comma at the end of "
            f"each row can create an empty extra column)."
        )

    X = df[numeric_features + categorical_features]
    y_raw = df[target_column]

    label_encoder = LabelEncoder()
    y = np.asarray(label_encoder.fit_transform(y_raw.astype(str))).astype(np.int64)
    n_classes = len(label_encoder.classes_)
    if n_classes < 2:
        raise FriendlyError(
            f"The target column '{target_column}' contains only one distinct value, so it can't be "
            f"classified. Please use a dataset whose target has at least two classes."
        )

    stratify = y if min(np.bincount(y)) >= 2 else None

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipeline = build_feature_pipeline(numeric_features, categorical_features, resolved)
    X_train = pipeline.fit_transform(X_train_raw)
    X_test = pipeline.transform(X_test_raw)

    # Ensure dense arrays (OneHotEncoder can return sparse matrices)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    return X_train, X_test, y_train, y_test, pipeline, label_encoder, n_classes, resolved