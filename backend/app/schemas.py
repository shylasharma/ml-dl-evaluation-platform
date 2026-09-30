from typing import List, Optional, Dict, Any
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


class PreprocessingConfig(BaseModel):
    """
    Explicit, reproducible preprocessing choices for one experiment.

    Defaults here intentionally match the platform's original hard-coded
    behaviour (median imputation / most-frequent imputation / StandardScaler /
    one-hot encoding / keep duplicates). A request that omits `preprocessing`
    entirely, or sends these defaults, produces identical results to before
    this feature existed -- see `resolve_preprocessing_config` in
    `app/ml/preprocessing.py` for how backward compatibility is enforced.
    """
    missing_numeric: str = Field(
        "median", description="How to fill missing numeric values: 'median' or 'mean'.")
    missing_categorical: str = Field(
        "most_frequent", description="How to fill missing categorical values: 'most_frequent'.")
    scaling: str = Field(
        "standard", description="Numeric feature scaling: 'none', 'standard', 'minmax', or 'robust'.")
    encoding: str = Field(
        "onehot", description="Categorical encoding method. Only 'onehot' is currently supported.")
    duplicates: str = Field(
        "keep", description="Whether to remove exact duplicate rows before splitting: 'keep' or 'remove'.")


class ExperimentConfig(BaseModel):
    dataset_id: int
    target_column: Optional[str] = None
    mode: str = Field("quick", description="'quick' or 'advanced'")
    models: List[str] = Field(..., min_items=1)
    imbalance_method: str = "none"
    compare_before_after: bool = False

    # Advanced options (ignored in quick mode, sensible defaults otherwise)
    test_size: float = 0.25
    cv_folds: int = 0  # 0 = no cross-validation, just a single train/test split
    cv_repeats: int = 1  # repeated k-fold when > 1 (only used if cv_folds > 0)
    random_state: int = 42
    scale_features: bool = True  # DEPRECATED: kept for backward compatibility; use `preprocessing.scaling` instead
    dl_epochs: int = 30
    dl_batch_size: int = 32
    primary_metric: str = "f1"

    # Optional, explicit preprocessing configuration (Phase 1). Omitting this
    # field entirely reproduces the platform's original behaviour exactly.
    preprocessing: Optional[PreprocessingConfig] = None


class ExperimentSummary(BaseModel):
    id: int
    name: str
    dataset_name: str
    status: str
    created_at: str
    config: Dict[str, Any]


class ExperimentResult(BaseModel):
    id: int
    name: str
    dataset_name: str
    status: str
    config: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    insights: Optional[List[str]] = None
    conclusion: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str