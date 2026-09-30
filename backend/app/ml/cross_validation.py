"""
Repeated stratified k-fold cross-validation.

Unlike the single train/test split path, this refits the ENTIRE preprocessing
pipeline (imputation, scaling, encoding) and the imbalance-handling technique
independently for every fold, using only that fold's training rows. This is
the correct way to avoid leakage under cross-validation: nothing computed on
a fold's test rows may influence how that fold's training rows are prepared.
"""
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from app.ml.preprocessing import build_feature_pipeline, DEFAULT_PREPROCESSING
from app.ml.imbalance import apply_imbalance_technique, ImbalanceError
from app.ml.train_ml import train_and_evaluate_ml, ModelTrainingError
from app.ml.train_dl import train_and_evaluate_dl
from app.ml.registry import ML_MODELS, DL_MODELS

CV_METRICS = ["accuracy", "precision", "recall", "specificity", "f1",
              "roc_auc", "pr_auc", "mcc", "balanced_accuracy", "g_mean"]


def cross_validate_model(
    model_key: str,
    X_df, y: np.ndarray,
    numeric_features: List[str], categorical_features: List[str],
    n_classes: int,
    imbalance_method: str,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    scale_features: bool = True,
    dl_epochs: int = 30,
    dl_batch_size: int = 32,
    preprocessing: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    # `preprocessing`, when given, must already be resolved/validated (see
    # `resolve_preprocessing_config`). Falls back to the legacy `scale_features`
    # boolean when omitted, for backward compatibility.
    resolved_preprocessing = preprocessing or {**DEFAULT_PREPROCESSING, "scaling": "standard" if scale_features else "none"}

    if model_key not in ML_MODELS and model_key not in DL_MODELS:
        return {"model_key": model_key, "model_label": model_key, "family": "unknown",
                "error": f"Unknown model '{model_key}'."}

    spec = ML_MODELS.get(model_key) or DL_MODELS.get(model_key)
    family = "ML" if model_key in ML_MODELS else "DL"

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    fold_scores: Dict[str, List[float]] = {m: [] for m in CV_METRICS}
    fold_errors: List[str] = []
    fold_confusion_sum = None
    fold_importances: List[np.ndarray] = []
    best_fold_result = None  # the fold whose ROC/PR curve we show as representative

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_df, y)):
        X_train_raw = X_df.iloc[train_idx]
        X_test_raw = X_df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Refit the whole preprocessing pipeline inside this fold, using only
        # this fold's training rows, exactly as before Phase 1 -- only the
        # *configuration* (which imputer/scaler to use) is now configurable.
        pipeline = build_feature_pipeline(numeric_features, categorical_features, resolved_preprocessing)
        try:
            X_train = pipeline.fit_transform(X_train_raw)
            X_test = pipeline.transform(X_test_raw)
        except Exception as exc:
            fold_errors.append(f"Fold {fold_idx + 1}: preprocessing failed ({exc})")
            continue
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()

        try:
            X_res, y_res, class_weight = apply_imbalance_technique(
                X_train, y_train, imbalance_method, random_state=random_state
            )
        except ImbalanceError as exc:
            fold_errors.append(f"Fold {fold_idx + 1}: {exc}")
            continue

        try:
            if family == "ML":
                result = train_and_evaluate_ml(
                    model_key, X_res, y_res, X_test, y_test,
                    n_classes=n_classes, class_weight=class_weight, random_state=random_state,
                )
            else:
                result = train_and_evaluate_dl(
                    model_key, X_res, y_res, X_test, y_test,
                    n_classes=n_classes, class_weight=class_weight,
                    epochs=dl_epochs, batch_size=dl_batch_size, random_state=random_state,
                )
        except ModelTrainingError as exc:
            fold_errors.append(f"Fold {fold_idx + 1}: {exc}")
            continue

        for m in CV_METRICS:
            val = result.get(m)
            if val is not None:
                fold_scores[m].append(val)

        cm = np.array(result.get("confusion_matrix") or [])
        if cm.size:
            fold_confusion_sum = cm if fold_confusion_sum is None else fold_confusion_sum + cm

        if result.get("feature_importance"):
            fold_importances.append(np.array(result["feature_importance"]))

        if best_fold_result is None or (result.get("f1") or 0) > (best_fold_result.get("f1") or 0):
            best_fold_result = result

    n_completed = len(fold_scores["f1"])
    if n_completed == 0:
        return {"model_key": model_key, "model_label": spec.label, "family": family,
                "error": "; ".join(fold_errors) or "All folds failed to train."}

    metrics_summary = {}
    for m in CV_METRICS:
        vals = fold_scores[m]
        metrics_summary[m] = (
            {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": [float(v) for v in vals]}
            if vals else None
        )

    avg_importance = (
        np.mean(np.vstack(fold_importances), axis=0).tolist() if fold_importances else None
    )

    # Flat, top-level metric fields so this result is drop-in compatible with
    # the single-split UI (dashboard, compare, model detail) — using the mean
    # across folds as "the" value, with full distributions kept alongside.
    flat = {m: (metrics_summary[m]["mean"] if metrics_summary[m] else None) for m in CV_METRICS}

    return {
        "model_key": model_key,
        "model_label": spec.label,
        "family": family,
        "training_time_sec": best_fold_result.get("training_time_sec") if best_fold_result else None,
        "feature_importance": avg_importance,
        "confusion_matrix": fold_confusion_sum.astype(int).tolist() if fold_confusion_sum is not None else None,
        "roc_curve": best_fold_result.get("roc_curve") if best_fold_result else None,
        "pr_curve": best_fold_result.get("pr_curve") if best_fold_result else None,
        **flat,
        "cv_details": {
            "n_folds_requested": n_splits * n_repeats,
            "n_folds_completed": n_completed,
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "fold_errors": fold_errors,
            "metrics": metrics_summary,
            "note": "Confusion matrix is summed across folds. ROC/PR curves and "
                    "feature importance shown are averaged/representative across folds, "
                    "not from a single held-out split.",
        },
    }


def cross_validate_experiment(
    X_df, y: np.ndarray,
    numeric_features: List[str], categorical_features: List[str],
    n_classes: int,
    model_keys: List[str],
    imbalance_method: str,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    scale_features: bool = True,
    dl_epochs: int = 30,
    dl_batch_size: int = 32,
    preprocessing: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    return [
        cross_validate_model(
            m, X_df, y, numeric_features, categorical_features, n_classes, imbalance_method,
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,
            scale_features=scale_features, dl_epochs=dl_epochs, dl_batch_size=dl_batch_size,
            preprocessing=preprocessing,
        )
        for m in model_keys
    ]