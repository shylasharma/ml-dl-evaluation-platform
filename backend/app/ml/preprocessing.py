"""
Preprocessing utilities.

Critical research-methodology rule enforced here: the train/test split
happens BEFORE any resampling technique (e.g. SMOTE) is applied, and
scaling/encoding is *fit* only on the training fold and re-applied
(transform-only) to the test fold. This avoids data leakage.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_feature_pipeline(numeric_features: List[str], categorical_features: List[str],
                            scale_features: bool = True) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_features:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
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
):
    """
    Returns X_train, X_test (dense numpy arrays), y_train, y_test (encoded
    integer labels), the fitted feature pipeline, the fitted label encoder,
    and the number of classes.
    """
    df = df.dropna(subset=[target_column]).reset_index(drop=True)

    X = df[numeric_features + categorical_features]
    y_raw = df[target_column]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.astype(str))
    n_classes = len(label_encoder.classes_)

    stratify = y if min(np.bincount(y)) >= 2 else None

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipeline = build_feature_pipeline(numeric_features, categorical_features, scale_features)
    X_train = pipeline.fit_transform(X_train_raw)
    X_test = pipeline.transform(X_test_raw)

    # Ensure dense arrays (OneHotEncoder can return sparse matrices)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    return X_train, X_test, y_train, y_test, pipeline, label_encoder, n_classes
