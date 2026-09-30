from typing import Any, Dict

from fastapi import APIRouter

from app.ml.registry import ML_MODELS, DL_MODELS, IMBALANCE_METHODS
from app.ml.recommender import recommend_models

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
def list_models():
    ml = [
        {
            "key": s.key,
            "label": s.label,
            "family": s.family,
            "description": s.description,
        }
        for s in ML_MODELS.values()
    ]
    dl = [
        {
            "key": s.key,
            "label": s.label,
            "family": s.family,
            "description": s.description,
        }
        for s in DL_MODELS.values()
    ]
    return {"ml_models": ml, "dl_models": dl}


@router.get("/imbalance-methods")
def list_imbalance_methods():
    return [
        {
            "key": s.key,
            "label": s.label,
            "description": s.description,
            "kind": s.kind,
        }
        for s in IMBALANCE_METHODS.values()
    ]


@router.get("/metrics")
def list_metrics():
    return [
        {"key": "accuracy", "label": "Accuracy", "imbalance_relevant": False},
        {"key": "precision", "label": "Precision", "imbalance_relevant": False},
        {"key": "recall", "label": "Recall / Sensitivity", "imbalance_relevant": True},
        {"key": "specificity", "label": "Specificity", "imbalance_relevant": False},
        {"key": "f1", "label": "F1 Score", "imbalance_relevant": True},
        {"key": "roc_auc", "label": "ROC-AUC", "imbalance_relevant": False},
        {"key": "pr_auc", "label": "PR-AUC", "imbalance_relevant": True},
        {"key": "mcc", "label": "Matthews Correlation Coefficient", "imbalance_relevant": True},
        {"key": "balanced_accuracy", "label": "Balanced Accuracy", "imbalance_relevant": True},
        {"key": "g_mean", "label": "G-Mean", "imbalance_relevant": True},
    ]


@router.post("/recommend-models")
def recommend_model_endpoint(payload: Dict[str, Any]):
    return recommend_models(payload)
