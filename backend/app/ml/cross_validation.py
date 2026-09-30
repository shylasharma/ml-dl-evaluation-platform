"""
Leakage-safe repeated stratified cross-validation.

Per fold:

    Train fold
        ↓
    Fit preprocessing
        ↓
    Transform train + validation
        ↓
    Fit feature selection on TRAIN ONLY
        ↓
    Transform train + validation
        ↓
    Apply imbalance handling to TRAIN ONLY
        ↓
    Train model
        ↓
    Evaluate on untouched validation fold

Feature selection is deliberately re-fitted independently for every fold.
"""

from typing import Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

from app.ml.preprocessing import build_feature_pipeline
from app.ml.feature_engineering import (
    resolve_feature_engineering_config,
    fit_transform_feature_engineering,
)
from app.ml.feature_selection import (
    resolve_feature_selection_config,
    fit_transform_feature_selection,
)
from app.ml.dimensionality_reduction import fit_transform_pca

from app.ml.imbalance import (
    apply_imbalance_technique,
    ImbalanceError,
)
from app.ml.train_ml import (
    train_and_evaluate_ml,
    ModelTrainingError,
)
from app.ml.train_dl import train_and_evaluate_dl
from app.ml.registry import (
    ML_MODELS,
    DL_MODELS,
)


# ============================================================================
# HELPERS
# ============================================================================

def _to_dense(X):
    """
    Convert sparse matrices to dense numpy arrays.
    """

    if hasattr(X, "toarray"):
        return X.toarray()

    return np.asarray(X)


def _aggregate_model_results(
    model_key: str,
    model_label: str,
    family: str,
    fold_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate numeric metrics across CV folds.

    The frontend expects the normal model-result structure, while the
    additional fold-level information is retained for research analysis.
    """

    successful = [
        result
        for result in fold_results
        if "error" not in result
    ]

    if not successful:

        first_error = (
            fold_results[0].get("error")
            if fold_results
            else "Model failed in all folds."
        )

        return {
            "model_key": model_key,
            "model_label": model_label,
            "family": family,
            "error": first_error,
            "fold_results": fold_results,
        }

    # Metrics produced by metrics.py.
    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
        "balanced_accuracy",
        "g_mean",
    ]

    aggregated = {}

    metric_std = {}

    for metric in metric_keys:

        values = []

        for result in successful:

            value = result.get(metric)

            if isinstance(value, (int, float)) and np.isfinite(value):
                values.append(float(value))

        if values:

            aggregated[metric] = round(
                float(np.mean(values)),
                6,
            )

            metric_std[metric] = round(
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
                6,
            )

        else:

            aggregated[metric] = None
            metric_std[metric] = None

    # Training time is averaged separately.
    training_times = [
        float(result["training_time_sec"])
        for result in successful
        if isinstance(
            result.get("training_time_sec"),
            (int, float),
        )
    ]

    if training_times:

        training_time = round(
            float(np.mean(training_times)),
            3,
        )

    else:

        training_time = None

    # Keep the model's feature importance when available.
    feature_importance = None

    importance_values = [
        result.get("feature_importance")
        for result in successful
        if result.get("feature_importance") is not None
    ]

    if importance_values:

        try:

            arrays = [
                np.asarray(value, dtype=float)
                for value in importance_values
            ]

            # Only average if dimensions match.
            if len({
                array.shape
                for array in arrays
            }) == 1:

                feature_importance = (
                    np.mean(
                        np.stack(arrays),
                        axis=0,
                    )
                    .tolist()
                )

        except Exception:

            feature_importance = None

    return {
        "model_key": model_key,
        "model_label": model_label,
        "family": family,

        **aggregated,

        "training_time_sec": training_time,

        "feature_importance": feature_importance,

        # Research information.
        "cv_mean": aggregated,
        "cv_std": metric_std,
        "fold_results": fold_results,
        "n_successful_folds": len(successful),
        "n_failed_folds": (
            len(fold_results)
            - len(successful)
        ),
    }


# ============================================================================
# MAIN CROSS-VALIDATION FUNCTION
# ============================================================================

def cross_validate_experiment(
    X_df: pd.DataFrame,
    y,
    numeric_features: List[str],
    categorical_features: List[str],
    n_classes: int,
    models_selected: List[str],
    imbalance_key: str,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    dl_epochs: int = 30,
    dl_batch_size: int = 32,
    preprocessing: Dict[str, Any] | None = None,
    feature_engineering: Dict[str, Any] | None = None,
    feature_selection: Dict[str, Any] | None = None,
    pca: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Run repeated stratified cross-validation.

    Parameters
    ----------
    X_df:
        Raw feature dataframe.

    y:
        Encoded target.

    numeric_features:
        Original numeric column names.

    categorical_features:
        Original categorical column names.

    n_classes:
        Number of target classes.

    models_selected:
        Models to evaluate.

    imbalance_key:
        Imbalance-handling method.

    preprocessing:
        Resolved preprocessing configuration.

    feature_engineering:
        Resolved feature-engineering configuration.

    feature_selection:
        Resolved feature-selection configuration.
    """

    preprocessing = preprocessing or {}
    feature_engineering = resolve_feature_engineering_config(
        {"feature_engineering": feature_engineering if feature_engineering else None}
    )
    pca = pca or {}

    # Resolve feature-selection configuration safely.
    feature_selection = resolve_feature_selection_config(
        {
            "feature_selection": (
                feature_selection
                if feature_selection
                else None
            )
        }
    )

    # ------------------------------------------------------------------------
    # CV splitter
    # ------------------------------------------------------------------------

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    # ------------------------------------------------------------------------
    # Store fold-level results for every model.
    # ------------------------------------------------------------------------

    fold_results_by_model = {
        model_key: []
        for model_key in models_selected
    }

    # ------------------------------------------------------------------------
    # Track feature-engineering / feature-selection information.
    # ------------------------------------------------------------------------

    feature_engineering_by_fold = []
    feature_selection_by_fold = []
    pca_by_fold = []

    # =========================================================================
    # FOLD LOOP
    # =========================================================================

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cv.split(X_df, y),
        start=1,
    ):

        # ---------------------------------------------------------------------
        # Raw fold data
        # ---------------------------------------------------------------------

        X_train_df = X_df.iloc[
            train_indices
        ].copy()

        X_validation_df = X_df.iloc[
            validation_indices
        ].copy()

        y_train = np.asarray(y)[
            train_indices
        ]

        y_validation = np.asarray(y)[
            validation_indices
        ]

        # ---------------------------------------------------------------------
        # PREPROCESSING
        # ---------------------------------------------------------------------
        #
        # Build a NEW preprocessing pipeline for every fold.
        #
        # This guarantees that:
        # - imputers learn only from fold training data
        # - scalers learn only from fold training data
        # - encoders learn only from fold training data
        #
        # Validation data is only transformed.

        pipeline = build_feature_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            preprocessing=preprocessing,
        )

        pipeline.fit(
            X_train_df
        )

        X_train = pipeline.transform(
            X_train_df
        )

        X_validation = pipeline.transform(
            X_validation_df
        )

        X_train = _to_dense(X_train)
        X_validation = _to_dense(X_validation)

        # ---------------------------------------------------------------------
        # FEATURE NAMES AFTER PREPROCESSING
        # ---------------------------------------------------------------------

        try:

            feature_names = list(
                pipeline.get_feature_names_out()
            )

        except Exception:

            feature_names = [
                f"feature_{index}"
                for index in range(
                    X_train.shape[1]
                )
            ]

        # ---------------------------------------------------------------------
        # FEATURE ENGINEERING
        # ---------------------------------------------------------------------
        #
        # CRITICAL:
        #
        # Feature engineering is fitted independently inside every fold.
        # The validation fold is transformed only after the training-fold
        # transformation has been learned. This keeps generated features
        # leakage-safe during cross-validation.

        feature_engineering_info = {
            "fold": fold_number,
            "enabled": bool(
                feature_engineering.get("enabled", False)
            ),
            "original_features": len(feature_names),
            "engineered_features": len(feature_names),
            "feature_names": feature_names,
        }

        if feature_engineering.get("enabled", False):

            (
                X_train,
                X_validation,
                feature_engineering_info,
            ) = fit_transform_feature_engineering(
                X_train=X_train,
                X_test=X_validation,
                feature_names=feature_names,
                config=feature_engineering,
                random_state=random_state + fold_number,
            )

            feature_engineering_info = {
                "fold": fold_number,
                **feature_engineering_info,
            }

            engineered_names = (
                feature_engineering_info.get("feature_names")
                or feature_engineering_info.get("engineered_feature_names")
            )

            if engineered_names:
                feature_names = engineered_names

        feature_engineering_by_fold.append(
            feature_engineering_info
        )

        # ---------------------------------------------------------------------
        # FEATURE SELECTION
        # ---------------------------------------------------------------------
        #
        # CRITICAL:
        #
        # The selector is fitted ONLY on X_train / y_train.
        #
        # It never sees X_validation while learning which features matter.
        #
        # This prevents feature-selection leakage.

        fold_feature_selection = {
            "fold": fold_number,
            "enabled": bool(
                feature_selection.get(
                    "enabled",
                    False,
                )
            ),
            "method": feature_selection.get(
                "method",
                "none",
            ),
            "original_features": len(
                feature_names
            ),
        }

        if feature_selection.get(
            "enabled",
            False,
        ):

            (
                X_train,
                X_validation,
                selection_info,
            ) = fit_transform_feature_selection(
                X_train=X_train,
                X_test=X_validation,
                y_train=y_train,
                feature_names=feature_names,
                config=feature_selection,
                random_state=random_state,
            )

            fold_feature_selection.update(
                selection_info
            )

        else:

            fold_feature_selection.update(
                {
                    "selected_features": len(
                        feature_names
                    ),
                    "selected_feature_names": feature_names,
                }
            )

        feature_selection_by_fold.append(
            fold_feature_selection
        )

        # ---------------------------------------------------------------------
        # PCA / DIMENSIONALITY REDUCTION
        # ---------------------------------------------------------------------
        #
        # PCA is fitted ONLY on this fold's training data.
        # The validation fold is transformed with that fitted PCA.
        # This prevents dimensionality-reduction leakage.
        #

        selected_feature_names = (
            fold_feature_selection.get(
                "selected_feature_names",
                feature_names,
            )
        )

        (
            X_train,
            X_validation,
            pca_info,
        ) = fit_transform_pca(
            X_train=X_train,
            X_test=X_validation,
            feature_names=selected_feature_names,
            config=pca,
            random_state=random_state + fold_number,
        )

        pca_by_fold.append(
            {
                "fold": fold_number,
                **pca_info,
            }
        )

        # =====================================================================
        # MODEL LOOP
        # =====================================================================

        for model_key in models_selected:

            # ---------------------------------------------------------------
            # Identify model
            # ---------------------------------------------------------------

            if model_key in ML_MODELS:

                model_spec = ML_MODELS[
                    model_key
                ]

                family = "ML"

            elif model_key in DL_MODELS:

                model_spec = DL_MODELS[
                    model_key
                ]

                family = "DL"

            else:

                fold_results_by_model[
                    model_key
                ].append(
                    {
                        "error": (
                            f"Unknown model "
                            f"'{model_key}'."
                        )
                    }
                )

                continue

            # ---------------------------------------------------------------
            # IMBALANCE HANDLING
            # ---------------------------------------------------------------
            #
            # Apply imbalance handling ONLY to the training fold.
            #
            # Validation data remains untouched.

            try:

                (
                    X_resampled,
                    y_resampled,
                    class_weight,
                ) = apply_imbalance_technique(
                    X_train,
                    y_train,
                    imbalance_key,
                    random_state=random_state,
                )

            except ImbalanceError as exc:

                fold_results_by_model[
                    model_key
                ].append(
                    {
                        "model_key": model_key,
                        "model_label": model_spec.label,
                        "family": family,
                        "error": str(exc),
                    }
                )

                continue

            # ---------------------------------------------------------------
            # TRAIN + EVALUATE
            # ---------------------------------------------------------------

            try:

                if family == "ML":

                    result = train_and_evaluate_ml(
                        model_key,
                        X_resampled,
                        y_resampled,
                        X_validation,
                        y_validation,
                        n_classes=n_classes,
                        class_weight=class_weight,
                        random_state=(
                            random_state
                            + fold_number
                        ),
                    )

                else:

                    result = train_and_evaluate_dl(
                        model_key,
                        X_resampled,
                        y_resampled,
                        X_validation,
                        y_validation,
                        n_classes=n_classes,
                        class_weight=class_weight,
                        epochs=dl_epochs,
                        batch_size=dl_batch_size,
                        random_state=(
                            random_state
                            + fold_number
                        ),
                    )

                # Add fold metadata.
                result["fold"] = fold_number

                fold_results_by_model[
                    model_key
                ].append(result)

            except ModelTrainingError as exc:

                fold_results_by_model[
                    model_key
                ].append(
                    {
                        "model_key": model_key,
                        "model_label": model_spec.label,
                        "family": family,
                        "fold": fold_number,
                        "error": str(exc),
                    }
                )

            except Exception as exc:

                fold_results_by_model[
                    model_key
                ].append(
                    {
                        "model_key": model_key,
                        "model_label": model_spec.label,
                        "family": family,
                        "fold": fold_number,
                        "error": (
                            f"Unexpected training "
                            f"error: {exc}"
                        ),
                    }
                )

    # =========================================================================
    # AGGREGATE RESULTS
    # =========================================================================

    final_results = []

    for model_key in models_selected:

        fold_results = fold_results_by_model[
            model_key
        ]

        if model_key in ML_MODELS:

            model_label = ML_MODELS[
                model_key
            ].label

            family = "ML"

        elif model_key in DL_MODELS:

            model_label = DL_MODELS[
                model_key
            ].label

            family = "DL"

        else:

            model_label = model_key
            family = "unknown"

        aggregated = _aggregate_model_results(
            model_key=model_key,
            model_label=model_label,
            family=family,
            fold_results=fold_results,
        )

        final_results.append(
            aggregated
        )

    # =========================================================================
    # GLOBAL CV METADATA
    # =========================================================================

    for result in final_results:

        result["cv_info"] = {
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "n_folds": (
                n_splits
                * n_repeats
            ),
        }

        result[
            "feature_engineering_by_fold"
        ] = feature_engineering_by_fold

        result[
            "feature_selection_by_fold"
        ] = feature_selection_by_fold

        result[
            "pca_by_fold"
        ] = pca_by_fold

    return final_results