from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class DatasetProfile(BaseModel):
    dataset_id: int
    name: str
    rows: int
    n_features: int
    target_column: str
    columns: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    missing_values: Dict[str, int]
    duplicate_rows: int
    is_binary: bool
    class_counts: Dict[str, int]
    majority_class: str
    minority_class: str
    imbalance_ratio: float
    is_imbalanced: bool
    warnings: List[str]
    recommendations: List[str]
    preview: List[Dict[str, Any]]


class ModelInfo(BaseModel):
    key: str
    label: str
    family: str  # "ML" or "DL"
    description: str


class ImbalanceMethodInfo(BaseModel):
    key: str
    label: str
    description: str


# ============================================================
# PREPROCESSING
# ============================================================

class PreprocessingConfig(BaseModel):
    """
    Explicit, reproducible preprocessing choices for one experiment.
    """

    missing_numeric: str = Field(
        "median",
        description=(
            "How to fill missing numeric values: "
            "'median' or 'mean'."
        ),
    )

    missing_categorical: str = Field(
        "most_frequent",
        description=(
            "How to fill missing categorical values: "
            "'most_frequent' or 'constant'."
        ),
    )

    scaling: str = Field(
        "standard",
        description=(
            "Numeric feature scaling: "
            "'none', 'standard', 'minmax', or 'robust'."
        ),
    )

    encoding: str = Field(
        "onehot",
        description=(
            "Categorical encoding method. "
            "Only 'onehot' is currently supported."
        ),
    )

    duplicates: str = Field(
        "keep",
        description=(
            "Whether to remove exact duplicate rows "
            "before splitting: 'keep' or 'remove'."
        ),
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

class FeatureEngineeringConfig(BaseModel):
    """
    Configuration for optional feature engineering.

    Feature engineering is applied after preprocessing and
    before feature selection and PCA.
    """

    enabled: bool = False

    polynomial: bool = False

    polynomial_degree: int = Field(
        2,
        ge=2,
        le=3,
        description="Polynomial feature degree.",
    )

    interactions: bool = False

    log_transform: bool = False

    ratio_features: bool = False

    max_interaction_features: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Maximum number of source features used "
            "for interaction/ratio generation."
        ),
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

class FeatureSelectionConfig(BaseModel):
    enabled: bool = False

    method: str = Field(
        "none",
        description=(
            "Feature selection method: none, correlation, chi2, "
            "anova, mutual_information, rfe, sequential, "
            "l1, tree_importance."
        ),
    )

    k: Optional[int] = Field(
        None,
        ge=1,
        description="Number of features to select.",
    )

    threshold: Optional[str] = Field(
        None,
        description="Threshold for supported embedded selectors.",
    )

    correlation_threshold: float = Field(
        0.90,
        gt=0.0,
        lt=1.0,
    )

    scoring: str = Field(
        "f1_weighted",
        description=(
            "Scoring metric used by wrapper/embedded selectors."
        ),
    )

    direction: str = Field(
        "forward",
        description="Sequential selection direction.",
    )

    step: int = Field(
        1,
        ge=1,
    )

    max_features: Optional[int] = Field(
        None,
        ge=1,
    )


# ============================================================
# PCA / DIMENSIONALITY REDUCTION
# ============================================================

class PCAConfig(BaseModel):
    enabled: bool = False

    mode: str = Field(
        "variance",
        description=(
            "PCA selection mode: variance or components."
        ),
    )

    variance: float = Field(
        0.95,
        gt=0.0,
        le=1.0,
        description=(
            "Target cumulative variance to retain."
        ),
    )

    n_components: Optional[int] = Field(
        None,
        ge=1,
        description="Number of PCA components.",
    )

# ============================================================
#  Hybridization
# ============================================================
class HybridizationConfig(BaseModel):
    enabled: bool = False

    method: str = Field(
        "none",
        description=(
            "Hybridization method: none, hard_voting, soft_voting, "
            "stacking, or blending"
        ),
    )

    base_models: List[str] = Field(
        default_factory=list,
        description="ML models used as base learners",
    )

    weights: Optional[List[float]] = Field(
        None,
        description=(
            "Optional weights for voting or blending. "
            "Must match the number of selected base models."
        ),
    )

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

class ExperimentConfig(BaseModel):
    dataset_id: int

    target_column: Optional[str] = None

    mode: str = Field(
        "quick",
        description="quick or advanced",
    )

    models: List[str] = Field(
        ...,
        min_items=1,
    )

    imbalance_method: str = "none"

    compare_before_after: bool = False

    test_size: float = 0.25

    cv_folds: int = 0

    cv_repeats: int = 1

    random_state: int = 42

    scale_features: bool = True

    dl_epochs: int = 30

    dl_batch_size: int = 32

    primary_metric: str = "f1"

    preprocessing: Optional[PreprocessingConfig] = None
    feature_engineering: Optional[FeatureEngineeringConfig] = None
    feature_selection: Optional[FeatureSelectionConfig] = None
    pca: Optional[PCAConfig] = None
    compare_pca: bool = False
    hybridization: Optional[HybridizationConfig] = None


# ============================================================
# EXPERIMENT SUMMARY
# ============================================================

class ExperimentSummary(BaseModel):
    id: int
    name: str
    dataset_name: str
    status: str
    created_at: str
    config: Dict[str, Any]


# ============================================================
# EXPERIMENT RESULT
# ============================================================

class ExperimentResult(BaseModel):
    id: int
    name: str
    dataset_name: str
    status: str
    config: Dict[str, Any]

    results: Optional[
        Dict[str, Any]
    ] = None

    insights: Optional[
        List[str]
    ] = None

    conclusion: Optional[str] = None

    error_message: Optional[str] = None

    created_at: str