"""
Runs a full experiment: preprocess -> split -> (per model) resample training
fold -> train -> evaluate -> aggregate -> generate insights & conclusion.

Train/test split happens once, before any resampling, and resampling is
re-applied per model only to that model's copy of the training fold — this
keeps every model's evaluation leakage-free and directly comparable.
"""
from typing import Dict, Any, List
import pandas as pd

from app.ml.dataset_analysis import profile_dataset
from app.ml.preprocessing import prepare_data, resolve_preprocessing_config, drop_duplicate_rows
from app.ml.imbalance import apply_imbalance_technique, ImbalanceError
from app.ml.train_ml import train_and_evaluate_ml, ModelTrainingError
from app.ml.train_dl import train_and_evaluate_dl
from app.ml.registry import ML_MODELS, DL_MODELS, IMBALANCE_METHODS
from app.ml.insights import generate_insights, generate_conclusion, generate_significance_insights
from app.ml.cross_validation import cross_validate_experiment
from app.utils.errors import FriendlyError
from app.ml.statistical_tests import run_significance_tests


def run_experiment(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    target_column = config.get("target_column")
    # Profiling runs on the original, un-deduplicated data so the profile's
    # `duplicate_rows` count always reflects what was actually in the upload,
    # regardless of what the user chooses to do about it below.
    profile = profile_dataset(df, target_hint=target_column)
    target_column = profile["target_column"]

    # Preprocessing config is resolved ONCE here (Phase 1) so every stage
    # downstream (single-split path, CV path) sees the same, already-validated
    # configuration. Later phases (feature engineering, feature selection,
    # PCA) should follow the same pattern: resolve their own config here,
    # apply whole-dataset/deterministic steps here, and pass the resolved
    # config down to both `_run_single_split_experiment` and `_run_cv_experiment`.
    preprocessing_cfg = resolve_preprocessing_config(config)

    # Duplicate-row removal is deterministic (no statistic is fit from the
    # data) and is applied to the WHOLE dataset before the split -- see
    # `drop_duplicate_rows` for why this order is leakage-safe rather than
    # leakage-prone.
    df, n_duplicates_removed = drop_duplicate_rows(df, preprocessing_cfg["duplicates"])

    cv_folds = config.get("cv_folds", 0) or 0
    if cv_folds and cv_folds >= 2:
        return _run_cv_experiment(df, config, profile, target_column, preprocessing_cfg, n_duplicates_removed)
    return _run_single_split_experiment(df, config, profile, target_column, preprocessing_cfg, n_duplicates_removed)


def _run_single_split_experiment(df, config, profile, target_column,
                                  preprocessing_cfg: Dict[str, str], n_duplicates_removed: int) -> Dict[str, Any]:
    numeric_features = profile["numeric_features"]
    categorical_features = profile["categorical_features"]

    test_size = config.get("test_size", 0.25)
    random_state = config.get("random_state", 42)
    imbalance_key = config.get("imbalance_method", "none")
    models_selected: List[str] = config["models"]
    dl_epochs = config.get("dl_epochs", 30)
    dl_batch_size = config.get("dl_batch_size", 32)
    compare_before_after = config.get("compare_before_after", False)
    primary_metric = config.get("primary_metric", "f1")

    if imbalance_key not in IMBALANCE_METHODS:
        raise ValueError(f"Unknown imbalance method '{imbalance_key}'.")
    imbalance_label = IMBALANCE_METHODS[imbalance_key].label

    # `preprocessing_cfg` is fit ONLY on X_train inside `prepare_data` (median/
    # mean/most-frequent statistics and the scaler's mean/std or min/max are
    # all learned from the training split only, then merely applied to the
    # test split) -- this behaviour is unchanged from before Phase 1.
    X_train, X_test, y_train, y_test, pipeline, label_encoder, n_classes, preprocessing_applied = prepare_data(
        df, target_column, numeric_features, categorical_features,
        test_size=test_size, random_state=random_state, preprocessing=preprocessing_cfg,
    )

    def _train_one(model_key: str, method: str):
        try:
            X_res, y_res, class_weight = apply_imbalance_technique(
                X_train, y_train, method, random_state=random_state
            )
        except ImbalanceError as exc:
            return {"model_key": model_key,
                    "model_label": ML_MODELS.get(model_key, DL_MODELS.get(model_key)).label,
                    "family": "ML" if model_key in ML_MODELS else "DL",
                    "error": str(exc)}

        try:
            if model_key in ML_MODELS:
                return train_and_evaluate_ml(
                    model_key, X_res, y_res, X_test, y_test,
                    n_classes=n_classes, class_weight=class_weight, random_state=random_state,
                )
            elif model_key in DL_MODELS:
                return train_and_evaluate_dl(
                    model_key, X_res, y_res, X_test, y_test,
                    n_classes=n_classes, class_weight=class_weight,
                    epochs=dl_epochs, batch_size=dl_batch_size, random_state=random_state,
                )
            else:
                return {"model_key": model_key, "model_label": model_key,
                        "family": "unknown", "error": f"Unknown model '{model_key}'."}
        except ModelTrainingError as exc:
            family = "ML" if model_key in ML_MODELS else "DL"
            label = ML_MODELS.get(model_key, DL_MODELS.get(model_key, None))
            return {"model_key": model_key,
                    "model_label": label.label if label else model_key,
                    "family": family, "error": str(exc)}

    results = [_train_one(m, imbalance_key) for m in models_selected]

    before_after = None
    if compare_before_after and imbalance_key != "none":
        before_after = {}
        for m in models_selected:
            before = _train_one(m, "none")
            after = next((r for r in results if r["model_key"] == m), None)
            before_after[m] = {
                "model_label": before.get("model_label", m),
                "before": before if "error" not in before else {},
                "after": after if after and "error" not in after else {},
            }

    insights = generate_insights(
        results, dataset_profile=profile,
        imbalance_method_label=imbalance_label,
        before_after=before_after,
    )
    conclusion = generate_conclusion(
        results, dataset_profile=profile,
        imbalance_method_label=imbalance_label,
        primary_metric=primary_metric,
    )

    return {
        "dataset_profile": profile,
        "imbalance_method": imbalance_key,
        "imbalance_method_label": imbalance_label,
        "n_classes": n_classes,
        "class_labels": [str(c) for c in label_encoder.classes_],
        "cv_enabled": False,
        "results": results,
        "before_after": before_after,
        "insights": insights,
        "conclusion": conclusion,
        # What was actually applied -- always present, even for legacy requests
        # that never sent a `preprocessing` block, so results are self-describing.
        "preprocessing": {**preprocessing_applied, "duplicate_rows_removed": n_duplicates_removed},
    }


def _run_cv_experiment(df, config, profile, target_column,
                        preprocessing_cfg: Dict[str, str], n_duplicates_removed: int) -> Dict[str, Any]:
    from sklearn.preprocessing import LabelEncoder

    numeric_features = profile["numeric_features"]
    categorical_features = profile["categorical_features"]

    random_state = config.get("random_state", 42)
    imbalance_key = config.get("imbalance_method", "none")
    models_selected: List[str] = config["models"]
    dl_epochs = config.get("dl_epochs", 30)
    dl_batch_size = config.get("dl_batch_size", 32)
    primary_metric = config.get("primary_metric", "f1")
    n_splits = int(config.get("cv_folds", 5))
    n_repeats = max(1, int(config.get("cv_repeats", 1)))

    if imbalance_key not in IMBALANCE_METHODS:
        raise ValueError(f"Unknown imbalance method '{imbalance_key}'.")
    imbalance_label = IMBALANCE_METHODS[imbalance_key].label

    clean_df = df.dropna(subset=[target_column]).reset_index(drop=True)
    if clean_df.empty:
        raise FriendlyError(f"The target column '{target_column}' has no values, so there is nothing to predict.")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(clean_df[target_column].astype(str))
    n_classes = len(label_encoder.classes_)
    if n_classes < 2:
        raise FriendlyError(f"The target column '{target_column}' contains only one distinct value.")
    X_df = clean_df[numeric_features + categorical_features]

    min_class_count = int(pd_value_counts_min(y))
    if min_class_count < n_splits:
        raise ValueError(
            f"Cannot run {n_splits}-fold cross-validation: the smallest class has only "
            f"{min_class_count} sample(s). Reduce the number of folds to at most {min_class_count}, "
            f"or use a single train/test split instead."
        )

    # Each fold refits the whole preprocessing pipeline on that fold's training
    # rows only (see `cross_validation.py`), using this same resolved config.
    results = cross_validate_experiment(
        X_df, y, numeric_features, categorical_features, n_classes,
        models_selected, imbalance_key,
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,
        dl_epochs=dl_epochs, dl_batch_size=dl_batch_size, preprocessing=preprocessing_cfg,
    )

    significance = run_significance_tests(results, metric=primary_metric)

    cv_info = {"n_splits": n_splits, "n_repeats": n_repeats,
               "n_folds_requested": n_splits * n_repeats}

    insights = generate_insights(
        results, dataset_profile=profile, imbalance_method_label=imbalance_label,
    )
    insights += generate_significance_insights(significance)

    conclusion = generate_conclusion(
        results, dataset_profile=profile, imbalance_method_label=imbalance_label,
        primary_metric=primary_metric, cv_info=cv_info, significance=significance,
    )

    return {
        "dataset_profile": profile,
        "imbalance_method": imbalance_key,
        "imbalance_method_label": imbalance_label,
        "n_classes": n_classes,
        "class_labels": [str(c) for c in label_encoder.classes_],
        "cv_enabled": True,
        "cv_info": cv_info,
        "results": results,
        "before_after": None,
        "significance": significance,
        "insights": insights,
        "conclusion": conclusion,
        "preprocessing": {**preprocessing_cfg, "duplicate_rows_removed": n_duplicates_removed},
    }


def pd_value_counts_min(y) -> int:
    import numpy as np
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())