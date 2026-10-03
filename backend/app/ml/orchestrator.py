"""
ML/DL Model Evaluation & Comparison Platform
Experiment Orchestrator

Pipeline:

Dataset
    ↓
Dataset Profiling
    ↓
Duplicate Handling
    ↓
Train/Test Split
    ↓
Preprocessing
    ↓
Feature Selection
    ↓
PCA / Dimensionality Reduction
    ↓
Imbalance Handling
    ↓
Model Training
    ↓
Evaluation
    ↓
Research Analysis
    ↓
Insights / Conclusion

Leakage-safety rules:

- Duplicate removal is deterministic and happens before splitting.
- Preprocessing is fitted only on training data.
- Feature selection is fitted only on training data.
- PCA is fitted only on training data.
- Test data is only transformed, never used for fitting.
- Resampling is applied only to training data.
- Cross-validation refits preprocessing and feature selection
  independently inside every fold.
"""

from typing import Dict, Any, List

import pandas as pd

from app.ml.dataset_analysis import profile_dataset

from app.ml.preprocessing import (
    prepare_data,
    resolve_preprocessing_config,
    drop_duplicate_rows,
)

from app.ml.feature_selection import (
    resolve_feature_selection_config,
    fit_transform_feature_selection,
)

from app.ml.dimensionality_reduction import (
    fit_transform_pca,
)

from app.ml.feature_engineering import (
    resolve_feature_engineering_config,
    fit_transform_feature_engineering,
)

from app.ml.imbalance import (
    apply_imbalance_technique,
    ImbalanceError,
)

from app.ml.train_ml import (
    train_and_evaluate_ml,
    ModelTrainingError,
)

from app.ml.train_dl import (
    train_and_evaluate_dl,
)

from app.ml.registry import (
    ML_MODELS,
    DL_MODELS,
    IMBALANCE_METHODS,
)

from app.ml.insights import (
    generate_insights,
    generate_conclusion,
    generate_significance_insights,
)

from app.ml.cross_validation import (
    cross_validate_experiment,
)

from app.utils.errors import (
    FriendlyError,
)

from app.ml.statistical_tests import (
    run_significance_tests,
)

from app.ml.research_analysis import (
    build_research_summary,
)


# ============================================================================
# CONFIGURATION RESOLVERS
# ============================================================================

def resolve_pca_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve PCA configuration into a plain dictionary.

    Supported configuration:

    {
        "pca": {
            "enabled": True,
            "mode": "variance",
            "variance": 0.95,
            "n_components": None
        }
    }

    OR:

    {
        "pca": {
            "enabled": True,
            "mode": "components",
            "variance": 0.95,
            "n_components": 10
        }
    }
    """

    cfg = config.get("pca")

    if not cfg:
        return {
            "enabled": False,
            "mode": "variance",
            "variance": 0.95,
            "n_components": None,
        }

    if hasattr(cfg, "model_dump"):
        cfg = cfg.model_dump()

    elif hasattr(cfg, "dict"):
        cfg = cfg.dict()

    cfg = cfg or {}

    return {
        "enabled": bool(
            cfg.get("enabled", False)
        ),
        "mode": cfg.get(
            "mode",
            "variance",
        ),
        "variance": cfg.get(
            "variance",
            0.95,
        ),
        "n_components": cfg.get(
            "n_components"
        ),
    }


# ============================================================================
# FEATURE ENGINEERING CONFIGURATION RESOLVER
# ============================================================================

def resolve_feature_engineering_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve feature-engineering configuration into a plain dictionary."""
    cfg = config.get("feature_engineering")

    if not cfg:
        return {
            "enabled": False,
            "polynomial": False,
            "polynomial_degree": 2,
            "interactions": False,
            "log_transform": False,
            "ratio_features": False,
            "max_interaction_features": None,
        }

    if hasattr(cfg, "model_dump"):
        cfg = cfg.model_dump()
    elif hasattr(cfg, "dict"):
        cfg = cfg.dict()

    cfg = cfg or {}

    return {
        "enabled": bool(cfg.get("enabled", False)),
        "polynomial": bool(cfg.get("polynomial", False)),
        "polynomial_degree": int(cfg.get("polynomial_degree", 2)),
        "interactions": bool(cfg.get("interactions", False)),
        "log_transform": bool(cfg.get("log_transform", False)),
        "ratio_features": bool(cfg.get("ratio_features", False)),
        "max_interaction_features": cfg.get("max_interaction_features"),
    }


def _build_pca_comparison(
    no_pca_result: Dict[str, Any],
    pca_result: Dict[str, Any],
    pca_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a model-by-model comparison between identical no-PCA and PCA runs."""

    no_pca_by_model = {
        row.get("model_key"): row
        for row in no_pca_result.get("results", [])
        if row.get("model_key")
    }
    pca_by_model = {
        row.get("model_key"): row
        for row in pca_result.get("results", [])
        if row.get("model_key")
    }

    model_keys = list(dict.fromkeys(
        list(no_pca_by_model.keys()) + list(pca_by_model.keys())
    ))

    metrics = [
        "accuracy", "precision", "recall", "specificity", "f1",
        "roc_auc", "pr_auc", "mcc", "balanced_accuracy", "g_mean",
        "training_time_sec",
    ]

    models = []
    for key in model_keys:
        before = no_pca_by_model.get(key, {})
        after = pca_by_model.get(key, {})
        metric_values = {}
        for metric in metrics:
            no_value = before.get(metric)
            pca_value = after.get(metric)
            metric_values[metric] = {
                "no_pca": no_value,
                "pca": pca_value,
                "delta": (
                    pca_value - no_value
                    if isinstance(no_value, (int, float))
                    and isinstance(pca_value, (int, float))
                    else None
                ),
            }
        models.append({
            "model_key": key,
            "model_label": (
                pca_by_model.get(key, {}).get("model_label")
                or no_pca_by_model.get(key, {}).get("model_label")
                or key
            ),
            "family": (
                pca_by_model.get(key, {}).get("family")
                or no_pca_by_model.get(key, {}).get("family")
            ),
            "no_pca_error": before.get("error"),
            "pca_error": after.get("error"),
            "metrics": metric_values,
        })

    return {
        "enabled": True,
        "pca_config": pca_config,
        "no_pca": {
            "enabled": False,
            "result_count": len(no_pca_result.get("results", [])),
        },
        "pca": {
            "enabled": True,
            "result_count": len(pca_result.get("results", [])),
            "pca_metadata": pca_result.get("pca"),
        },
        "models": models,
        "metric_names": metrics,
        "comparison_note": (
            "Both branches use the same dataset, preprocessing, feature engineering, "
            "feature selection, imbalance strategy, models, random state and evaluation "
            "settings. Only PCA is changed."
        ),
    }


# ============================================================================
# MAIN EXPERIMENT ENTRY POINT
# ============================================================================

def run_experiment(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the complete ML/DL experiment pipeline.

    Current stages:

        Dataset profiling
        Duplicate handling
        Preprocessing
        Feature selection
        PCA
        Imbalance handling
        ML/DL training
        Evaluation
        Research analysis

    Future stages:

        Feature engineering
        Model recommendation
    """

    # ------------------------------------------------------------------------
    # 1. DATASET PROFILING
    # ------------------------------------------------------------------------

    target_column = config.get(
        "target_column"
    )

    profile = profile_dataset(
        df,
        target_hint=target_column,
    )

    target_column = profile[
        "target_column"
    ]

    # ------------------------------------------------------------------------
    # 2. RESOLVE CONFIGURATION
    # ------------------------------------------------------------------------

    preprocessing_cfg = (
        resolve_preprocessing_config(
            config
        )
    )

    feature_selection_cfg = (
        resolve_feature_selection_config(
            config
        )
    )

    feature_engineering_cfg = resolve_feature_engineering_config(
        config
    )

    pca_cfg = resolve_pca_config(
        config
    )

    # ------------------------------------------------------------------------
    # OPTIONAL PCA VS NO-PCA COMPARISON
    # ------------------------------------------------------------------------
    # Run the exact same experiment configuration twice, changing only the
    # dimensionality-reduction stage. Nested runs disable comparison so this
    # block cannot recurse. This keeps the comparison fair and preserves the
    # existing leakage-safe pipeline in both branches.
    compare_pca = bool(config.get("compare_pca", False))
    if compare_pca:
        base_config = dict(config)
        base_config["compare_pca"] = False

        no_pca_config = dict(base_config)
        no_pca_config["pca"] = {
            "enabled": False,
            "mode": "variance",
            "variance": 0.95,
            "n_components": None,
        }

        pca_compare_config = dict(base_config)
        requested_pca = dict(pca_cfg)
        requested_pca["enabled"] = True
        pca_compare_config["pca"] = requested_pca

        no_pca_result = run_experiment(
            df.copy(), no_pca_config
        )
        pca_result = run_experiment(
            df.copy(), pca_compare_config
        )

        comparison = _build_pca_comparison(
            no_pca_result,
            pca_result,
            requested_pca,
        )

        # Use the explicitly requested PCA state as the primary result while
        # attaching the paired comparison for the dashboard/report.
        primary = (
            pca_result
            if pca_cfg.get("enabled", False)
            else no_pca_result
        )
        primary["pca_comparison"] = comparison
        primary["compare_pca"] = True
        return primary

    # ------------------------------------------------------------------------
    # 3. DUPLICATE HANDLING
    # ------------------------------------------------------------------------

    # Duplicate removal is deterministic.
    #
    # It does not learn any statistics from
    # the dataset, therefore it is safe to
    # perform before train/test splitting.

    df, n_duplicates_removed = (
        drop_duplicate_rows(
            df,
            preprocessing_cfg[
                "duplicates"
            ],
        )
    )

    # ------------------------------------------------------------------------
    # 4. CHOOSE SINGLE SPLIT OR CV
    # ------------------------------------------------------------------------

    cv_folds = config.get(
        "cv_folds",
        0,
    ) or 0

    if cv_folds and cv_folds >= 2:

        return _run_cv_experiment(
            df=df,
            config=config,
            profile=profile,
            target_column=target_column,
            preprocessing_cfg=(
                preprocessing_cfg
            ),
            feature_selection_cfg=(
                feature_selection_cfg
            ),
            feature_engineering_cfg=(
                feature_engineering_cfg
            ),
            pca_cfg=pca_cfg,
            n_duplicates_removed=(
                n_duplicates_removed
            ),
        )

    return _run_single_split_experiment(
        df=df,
        config=config,
        profile=profile,
        target_column=target_column,
        preprocessing_cfg=(
            preprocessing_cfg
        ),
        feature_selection_cfg=(
            feature_selection_cfg
        ),
        feature_engineering_cfg=(
            feature_engineering_cfg
        ),
        pca_cfg=pca_cfg,
        n_duplicates_removed=(
            n_duplicates_removed
        ),
    )


# ============================================================================
# SINGLE TRAIN / TEST EXPERIMENT
# ============================================================================

def _run_single_split_experiment(
    df,
    config,
    profile,
    target_column,
    preprocessing_cfg: Dict[str, Any],
    feature_selection_cfg: Dict[str, Any],
    feature_engineering_cfg: Dict[str, Any],
    pca_cfg: Dict[str, Any],
    n_duplicates_removed: int,
) -> Dict[str, Any]:

    numeric_features = profile[
        "numeric_features"
    ]

    categorical_features = profile[
        "categorical_features"
    ]

    # ------------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------------

    test_size = config.get(
        "test_size",
        0.25,
    )

    random_state = config.get(
        "random_state",
        42,
    )

    imbalance_key = config.get(
        "imbalance_method",
        "none",
    )

    models_selected: List[str] = (
        config["models"]
    )

    dl_epochs = config.get(
        "dl_epochs",
        30,
    )

    dl_batch_size = config.get(
        "dl_batch_size",
        32,
    )

    compare_before_after = config.get(
        "compare_before_after",
        False,
    )

    primary_metric = config.get(
        "primary_metric",
        "f1",
    )

    # ------------------------------------------------------------------------
    # Validate imbalance method
    # ------------------------------------------------------------------------

    if imbalance_key not in IMBALANCE_METHODS:

        raise ValueError(
            f"Unknown imbalance method "
            f"'{imbalance_key}'."
        )

    imbalance_label = (
        IMBALANCE_METHODS[
            imbalance_key
        ].label
    )

    # ------------------------------------------------------------------------
    # 5. PREPROCESSING
    # ------------------------------------------------------------------------

    # prepare_data performs:
    #
    # Train/test split
    # ↓
    # Fit preprocessing on X_train
    # ↓
    # Transform X_train
    # ↓
    # Transform X_test
    #
    # No test-set statistics are used
    # during fitting.

    (
        X_train,
        X_test,
        y_train,
        y_test,
        pipeline,
        label_encoder,
        n_classes,
        preprocessing_applied,
        feature_names,
    ) = prepare_data(
        df,
        target_column,
        numeric_features,
        categorical_features,
        test_size=test_size,
        random_state=random_state,
        preprocessing=preprocessing_cfg,
    )

    # ------------------------------------------------------------------------
    # 6. FEATURE SELECTION
    # ------------------------------------------------------------------------

    original_feature_count = len(
        feature_names
    )

    feature_selection_applied = {
        "enabled": False,
        "method": "none",
        "original_features": (
            original_feature_count
        ),
        "selected_features": (
            original_feature_count
        ),
        "selected_feature_names": (
            feature_names
        ),
    }

    if feature_selection_cfg.get(
        "enabled",
        False,
    ):

        (
            X_train,
            X_test,
            feature_selection_applied,
        ) = fit_transform_feature_selection(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            feature_names=feature_names,
            config=feature_selection_cfg,
            random_state=random_state,
        )

        # Update feature names when
        # feature-selection returns them.

        selected_names = (
            feature_selection_applied.get(
                "selected_feature_names"
            )
        )

        if selected_names:
            feature_names = (
                selected_names
            )

    # ------------------------------------------------------------------------
    # 7. FEATURE ENGINEERING
    # ------------------------------------------------------------------------

    # Feature engineering is fitted ONLY on the training data.
    # X_test is transformed using the fitted engineering configuration.
    # This keeps the stage leakage-safe.

    feature_engineering_applied = {
        "enabled": False,
        "polynomial": False,
        "polynomial_degree": 2,
        "interactions": False,
        "log_transform": False,
        "ratio_features": False,
        "max_interaction_features": None,
        "original_features": len(feature_names),
        "engineered_features": len(feature_names),
        "feature_names": feature_names,
    }

    (
        X_train,
        X_test,
        feature_engineering_applied,
    ) = fit_transform_feature_engineering(
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        config=feature_engineering_cfg,
        random_state=random_state,
    )

    engineered_names = feature_engineering_applied.get(
        "feature_names"
    ) or feature_engineering_applied.get(
        "engineered_feature_names"
    )

    if engineered_names:
        feature_names = engineered_names

    # ------------------------------------------------------------------------
    # 8. PCA / DIMENSIONALITY REDUCTION
    # ------------------------------------------------------------------------

    # IMPORTANT:
    #
    # PCA is fitted ONLY on X_train.
    #
    # X_test is transformed using the
    # already-fitted PCA object.
    #
    # Therefore PCA cannot learn anything
    # from the test set.

    (
        X_train,
        X_test,
        pca_applied,
    ) = fit_transform_pca(
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        config=pca_cfg,
        random_state=random_state,
    )

    if pca_applied.get(
        "enabled",
        False,
    ):

        feature_names = (
            pca_applied.get(
                "component_names",
                feature_names,
            )
        )

    # ------------------------------------------------------------------------
    # Helper: train one model
    # ------------------------------------------------------------------------

    def _train_one(
        model_key: str,
        method: str,
    ):

        # ---------------------------------------------------------------
        # 8. IMBALANCE HANDLING
        # ---------------------------------------------------------------

        # Resampling is performed ONLY on
        # training data.
        #
        # X_test and y_test remain untouched.

        try:

            X_res, y_res, class_weight = (
                apply_imbalance_technique(
                    X_train,
                    y_train,
                    method,
                    random_state=random_state,
                )
            )

        except ImbalanceError as exc:

            model_spec = ML_MODELS.get(
                model_key,
                DL_MODELS.get(model_key),
            )

            return {
                "model_key": model_key,

                "model_label": (
                    model_spec.label
                    if model_spec
                    else model_key
                ),

                "family": (
                    "ML"
                    if model_key in ML_MODELS
                    else "DL"
                ),

                "error": str(exc),
            }

        # ---------------------------------------------------------------
        # 9. MODEL TRAINING + EVALUATION
        # ---------------------------------------------------------------

        try:

            if model_key in ML_MODELS:

                result = (
                    train_and_evaluate_ml(
                        model_key,
                        X_res,
                        y_res,
                        X_test,
                        y_test,
                        n_classes=n_classes,
                        class_weight=(
                            class_weight
                        ),
                        random_state=(
                            random_state
                        ),
                    )
                )

            elif model_key in DL_MODELS:

                result = (
                    train_and_evaluate_dl(
                        model_key,
                        X_res,
                        y_res,
                        X_test,
                        y_test,
                        n_classes=n_classes,
                        class_weight=(
                            class_weight
                        ),
                        epochs=dl_epochs,
                        batch_size=(
                            dl_batch_size
                        ),
                        random_state=(
                            random_state
                        ),
                    )
                )

            else:

                return {
                    "model_key": model_key,
                    "model_label": model_key,
                    "family": "unknown",
                    "error": (
                        f"Unknown model "
                        f"'{model_key}'."
                    ),
                }

            return result

        except ModelTrainingError as exc:

            family = (
                "ML"
                if model_key in ML_MODELS
                else "DL"
            )

            label = ML_MODELS.get(
                model_key,
                DL_MODELS.get(model_key),
            )

            return {
                "model_key": model_key,

                "model_label": (
                    label.label
                    if label
                    else model_key
                ),

                "family": family,

                "error": str(exc),
            }

    # ------------------------------------------------------------------------
    # 10. TRAIN ALL SELECTED MODELS
    # ------------------------------------------------------------------------

    results = [
        _train_one(
            model_key=model_key,
            method=imbalance_key,
        )
        for model_key in models_selected
    ]

    # ------------------------------------------------------------------------
    # 11. BEFORE / AFTER IMBALANCE COMPARISON
    # ------------------------------------------------------------------------

    before_after = None

    if (
        compare_before_after
        and imbalance_key != "none"
    ):

        before_after = {}

        for model_key in models_selected:

            before = _train_one(
                model_key,
                "none",
            )

            after = next(
                (
                    result
                    for result in results
                    if result["model_key"]
                    == model_key
                ),
                None,
            )

            before_after[
                model_key
            ] = {

                "model_label": before.get(
                    "model_label",
                    model_key,
                ),

                "before": (
                    before
                    if "error" not in before
                    else {}
                ),

                "after": (
                    after
                    if after
                    and "error" not in after
                    else {}
                ),
            }

    # ------------------------------------------------------------------------
    # 12. AUTOMATIC INSIGHTS
    # ------------------------------------------------------------------------

    insights = generate_insights(
        results,
        dataset_profile=profile,
        imbalance_method_label=(
            imbalance_label
        ),
        before_after=before_after,
    )

    # ------------------------------------------------------------------------
    # 13. AUTOMATIC CONCLUSION
    # ------------------------------------------------------------------------

    conclusion = generate_conclusion(
        results,
        dataset_profile=profile,
        imbalance_method_label=(
            imbalance_label
        ),
        primary_metric=primary_metric,
    )

    # ------------------------------------------------------------------------
    # 14. RESEARCH SUMMARY
    # ------------------------------------------------------------------------

    research_summary = (
        build_research_summary(
            results=results,
            before_after=before_after,
            dataset_profile=profile,
            imbalance_method=(
                imbalance_label
            ),
            random_state=random_state,
        )
    )

    # ------------------------------------------------------------------------
    # 15. FINAL RESULT
    # ------------------------------------------------------------------------

    return {

        "dataset_profile": profile,

        "imbalance_method": imbalance_key,

        "imbalance_method_label": (
            imbalance_label
        ),

        "n_classes": n_classes,

        "class_labels": [
            str(c)
            for c in label_encoder.classes_
        ],

        "cv_enabled": False,

        "results": results,

        "before_after": before_after,

        "research_summary": (
            research_summary
        ),

        "insights": insights,

        "conclusion": conclusion,

        # --------------------------------------------------------------
        # Preprocessing information
        # --------------------------------------------------------------

        "preprocessing": {
            **preprocessing_applied,
            "duplicate_rows_removed": (
                n_duplicates_removed
            ),
        },

        # --------------------------------------------------------------
        # Feature selection information
        # --------------------------------------------------------------

        "feature_selection": (
            feature_selection_applied
        ),

        # --------------------------------------------------------------
        # Feature engineering information
        # --------------------------------------------------------------

        "feature_engineering": feature_engineering_applied,

        # --------------------------------------------------------------
        # PCA information
        # --------------------------------------------------------------

        "pca": pca_applied,

    }


# ============================================================================
# CROSS-VALIDATION EXPERIMENT
# ============================================================================

def _run_cv_experiment(
    df,
    config,
    profile,
    target_column,
    preprocessing_cfg: Dict[str, Any],
    feature_selection_cfg: Dict[str, Any],
    feature_engineering_cfg: Dict[str, Any],
    pca_cfg: Dict[str, Any],
    n_duplicates_removed: int,
) -> Dict[str, Any]:

    from sklearn.preprocessing import (
        LabelEncoder,
    )

    numeric_features = profile[
        "numeric_features"
    ]

    categorical_features = profile[
        "categorical_features"
    ]

    random_state = config.get(
        "random_state",
        42,
    )

    imbalance_key = config.get(
        "imbalance_method",
        "none",
    )

    models_selected: List[str] = (
        config["models"]
    )

    dl_epochs = config.get(
        "dl_epochs",
        30,
    )

    dl_batch_size = config.get(
        "dl_batch_size",
        32,
    )

    primary_metric = config.get(
        "primary_metric",
        "f1",
    )

    n_splits = int(
        config.get(
            "cv_folds",
            5,
        )
    )

    n_repeats = max(
        1,
        int(
            config.get(
                "cv_repeats",
                1,
            )
        ),
    )

    # ------------------------------------------------------------------------
    # Validate imbalance method
    # ------------------------------------------------------------------------

    if imbalance_key not in IMBALANCE_METHODS:

        raise ValueError(
            f"Unknown imbalance method "
            f"'{imbalance_key}'."
        )

    imbalance_label = (
        IMBALANCE_METHODS[
            imbalance_key
        ].label
    )

    # ------------------------------------------------------------------------
    # Clean target
    # ------------------------------------------------------------------------

    clean_df = (
        df
        .dropna(
            subset=[target_column]
        )
        .reset_index(drop=True)
    )

    if clean_df.empty:

        raise FriendlyError(
            f"The target column "
            f"'{target_column}' has no "
            "values, so there is nothing "
            "to predict."
        )

    # ------------------------------------------------------------------------
    # Encode target
    # ------------------------------------------------------------------------

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        clean_df[
            target_column
        ].astype(str)
    )

    n_classes = len(
        label_encoder.classes_
    )

    if n_classes < 2:

        raise FriendlyError(
            f"The target column "
            f"'{target_column}' contains "
            "only one distinct value."
        )

    X_df = clean_df[
        numeric_features
        + categorical_features
    ]

    # ------------------------------------------------------------------------
    # Validate number of folds
    # ------------------------------------------------------------------------

    min_class_count = int(
        pd_value_counts_min(y)
    )

    if min_class_count < n_splits:

        raise ValueError(
            f"Cannot run {n_splits}-fold "
            "cross-validation: the smallest "
            "class has only "
            f"{min_class_count} sample(s). "
            f"Reduce the number of folds to "
            f"at most {min_class_count}, or "
            "use a single train/test split "
            "instead."
        )

    # ------------------------------------------------------------------------
    # CROSS-VALIDATION
    # ------------------------------------------------------------------------

    # Feature selection is already supported
    # by the CV layer.
    #
    # PCA must follow the same leakage-safe
    # per-fold architecture.
    #
    # Therefore PCA configuration is passed
    # into cross_validate_experiment.

    results = cross_validate_experiment(
        X_df,
        y,
        numeric_features,
        categorical_features,
        n_classes,
        models_selected,
        imbalance_key,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        dl_epochs=dl_epochs,
        dl_batch_size=dl_batch_size,
        preprocessing=preprocessing_cfg,
        feature_engineering=(
            feature_engineering_cfg
        ),
        feature_selection=(
            feature_selection_cfg
        ),
        pca=pca_cfg,
    )

    # ------------------------------------------------------------------------
    # STATISTICAL SIGNIFICANCE
    # ------------------------------------------------------------------------

    significance = run_significance_tests(
        results,
        metric=primary_metric,
    )

    cv_info = {
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "n_folds_requested": (
            n_splits * n_repeats
        ),
    }

    # ------------------------------------------------------------------------
    # INSIGHTS
    # ------------------------------------------------------------------------

    insights = generate_insights(
        results,
        dataset_profile=profile,
        imbalance_method_label=(
            imbalance_label
        ),
    )

    insights += (
        generate_significance_insights(
            significance
        )
    )

    # ------------------------------------------------------------------------
    # CONCLUSION
    # ------------------------------------------------------------------------

    conclusion = generate_conclusion(
        results,
        dataset_profile=profile,
        imbalance_method_label=(
            imbalance_label
        ),
        primary_metric=primary_metric,
        cv_info=cv_info,
        significance=significance,
    )

    # ------------------------------------------------------------------------
    # RESEARCH SUMMARY
    # ------------------------------------------------------------------------

    research_summary = (
        build_research_summary(
            results=results,
            before_after=None,
            dataset_profile=profile,
            imbalance_method=(
                imbalance_label
            ),
            random_state=random_state,
        )
    )

    # ------------------------------------------------------------------------
    # FINAL CV RESULT
    # ------------------------------------------------------------------------

    return {

        "dataset_profile": profile,

        "imbalance_method": imbalance_key,

        "imbalance_method_label": (
            imbalance_label
        ),

        "n_classes": n_classes,

        "class_labels": [
            str(c)
            for c in label_encoder.classes_
        ],

        "cv_enabled": True,

        "cv_info": cv_info,

        "results": results,

        "before_after": None,

        "significance": significance,

        "research_summary": (
            research_summary
        ),

        "insights": insights,

        "conclusion": conclusion,

        "preprocessing": {
            **preprocessing_cfg,
            "duplicate_rows_removed": (
                n_duplicates_removed
            ),
        },

        "feature_selection": (
            feature_selection_cfg
        ),

        "feature_engineering": feature_engineering_cfg,

        "pca": pca_cfg,

    }


# ============================================================================
# UTILITY
# ============================================================================

def pd_value_counts_min(y) -> int:
    """
    Return the number of samples
    in the smallest class.
    """

    import numpy as np

    _, counts = np.unique(
        y,
        return_counts=True,
    )

    return int(
        counts.min()
    )