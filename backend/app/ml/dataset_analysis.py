"""
Computes real, data-driven statistics about an uploaded dataset:
row/feature counts, missing values, duplicates, feature types, target
detection, class distribution, and imbalance ratio.

Nothing here is hard-coded or assumed — every number is calculated from the
actual dataframe.
"""
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd


COMMON_TARGET_NAMES = [
    "target", "label", "class", "y", "outcome", "fraud", "is_fraud",
    "diagnosis", "churn", "default",
]


def json_safe_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Rows as plain dicts with NaN / +-inf replaced by None so they serialize to valid JSON."""
    safe = frame.replace([np.inf, -np.inf], np.nan).astype(object)
    safe = safe.where(pd.notnull(safe), None)
    return safe.to_dict(orient="records")


def load_clean_csv(path: str) -> pd.DataFrame:
    """
    Read a CSV and drop fully-empty columns/rows. Excel-style exports often end every
    row with a trailing comma, which pandas turns into an all-empty 'Unnamed: N' column
    that would otherwise be mistaken for the target.
    """
    df = pd.read_csv(path)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df.reset_index(drop=True)


def _has_classes(df: pd.DataFrame, col: str) -> bool:
    """A usable target has at least 2 distinct non-empty values."""
    return df[col].dropna().nunique() >= 2


def detect_target_column(df: pd.DataFrame, hint: Optional[str] = None) -> str:
    if hint and hint in df.columns:
        return hint
    lower_map = {c.lower(): c for c in df.columns}
    for name in COMMON_TARGET_NAMES:
        if name in lower_map and _has_classes(df, lower_map[name]):
            return lower_map[name]
    # Fall back: the last column that actually contains at least 2 distinct values
    for col in reversed(list(df.columns)):
        if _has_classes(df, col):
            return col
    return df.columns[-1]


def profile_dataset(df: pd.DataFrame, target_hint: Optional[str] = None) -> Dict[str, Any]:
    target_col = detect_target_column(df, target_hint)

    n_rows, n_cols_total = df.shape
    feature_cols = [c for c in df.columns if c != target_col]
    n_features = len(feature_cols)

    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    missing_values = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum() > 0}
    duplicate_rows = int(df.duplicated().sum())

    class_counts_series = df[target_col].value_counts(dropna=False)
    class_counts = {str(k): int(v) for k, v in class_counts_series.items()}
    n_classes = len(class_counts)
    is_binary = n_classes == 2

    sorted_classes = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)
    majority_class, majority_count = sorted_classes[0]
    minority_class, minority_count = sorted_classes[-1]
    imbalance_ratio = round(majority_count / minority_count, 2) if minority_count > 0 else float("inf")

    # A dataset is flagged as imbalanced when the majority class is at least
    # ~1.5x the minority class — a common, defensible rule of thumb — rather
    # than assuming every dataset is imbalanced.
    is_imbalanced = imbalance_ratio >= 1.5

    warnings: List[str] = []
    recommendations: List[str] = []

    if is_imbalanced:
        warnings.append(
            f"This dataset is imbalanced (majority:minority = {imbalance_ratio}:1)."
        )
        recommendations.append(
            "The dataset contains class imbalance. Consider evaluating imbalance-handling techniques."
        )
        recommendations.append(
            "Accuracy alone may be misleading here — Recall, F1, PR-AUC, and MCC may offer more useful signal."
        )
    else:
        warnings.append("Class distribution appears reasonably balanced.")

    if minority_count < 50:
        recommendations.append(
            f"The minority class contains only {minority_count} samples, which may limit model reliability."
        )

    if missing_values:
        total_missing = sum(missing_values.values())
        warnings.append(f"{total_missing} missing values detected across {len(missing_values)} column(s).")

    if duplicate_rows > 0:
        warnings.append(f"{duplicate_rows} duplicate row(s) detected.")

    if n_classes > 2:
        recommendations.append(
            f"This is a multiclass problem ({n_classes} classes); metric averaging (weighted/macro) is used."
        )

    preview = json_safe_records(df.head(10))

    return {
        "rows": int(n_rows),
        "n_features": int(n_features),
        "target_column": target_col,
        "columns": list(df.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "is_binary": is_binary,
        "n_classes": n_classes,
        "class_counts": class_counts,
        "majority_class": str(majority_class),
        "minority_class": str(minority_class),
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": is_imbalanced,
        "warnings": warnings,
        "recommendations": recommendations,
        "preview": preview,
    }