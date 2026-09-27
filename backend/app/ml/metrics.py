"""
Computes the full metric suite this platform focuses on: metrics that remain
informative under class imbalance, not just accuracy.
"""
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix, roc_curve, precision_recall_curve,
)


def _specificity_from_confusion(cm: np.ndarray) -> float:
    # Multiclass-safe macro specificity: average of per-class TN / (TN + FP)
    n = cm.shape[0]
    specs = []
    total = cm.sum()
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs))


def _g_mean(recall: float, specificity: float) -> float:
    return float(np.sqrt(max(recall, 0) * max(specificity, 0)))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_classes: int = 2,
) -> Dict[str, Any]:
    average = "binary" if n_classes <= 2 else "weighted"

    cm = confusion_matrix(y_true, y_pred)
    specificity = _specificity_from_confusion(cm)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "g_mean": _g_mean(recall, specificity),
        "confusion_matrix": cm.tolist(),
    }

    roc_curve_points = None
    pr_curve_points = None
    roc_auc = None
    pr_auc = None

    if y_proba is not None:
        try:
            if n_classes <= 2:
                proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                roc_auc = float(roc_auc_score(y_true, proba_pos))
                pr_auc = float(average_precision_score(y_true, proba_pos))
                fpr, tpr, _ = roc_curve(y_true, proba_pos)
                prec, rec, _ = precision_recall_curve(y_true, proba_pos)
                # Downsample curve points for compact JSON payloads
                roc_curve_points = _downsample(list(zip(fpr.tolist(), tpr.tolist())))
                pr_curve_points = _downsample(list(zip(rec.tolist(), prec.tolist())))
            else:
                roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
                pr_auc = float(average_precision_score(y_true, y_proba, average="weighted"))
        except Exception:
            pass  # ROC/PR AUC can fail for degenerate predictions; leave as None

    metrics["roc_auc"] = roc_auc
    metrics["pr_auc"] = pr_auc
    metrics["roc_curve"] = roc_curve_points
    metrics["pr_curve"] = pr_curve_points

    return metrics


def _downsample(points, max_points: int = 60):
    if len(points) <= max_points:
        return points
    step = len(points) / max_points
    return [points[int(i * step)] for i in range(max_points)]
