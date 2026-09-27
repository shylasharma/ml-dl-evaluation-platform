import time
from typing import Optional, Dict, Any
import numpy as np

from app.ml.metrics import compute_metrics
from app.ml.registry import ML_MODELS


class ModelTrainingError(Exception):
    pass


def train_and_evaluate_ml(
    model_key: str,
    X_train, y_train, X_test, y_test,
    n_classes: int,
    class_weight: Optional[dict] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    if model_key not in ML_MODELS:
        raise ModelTrainingError(f"Unknown ML model '{model_key}'.")

    spec = ML_MODELS[model_key]

    try:
        model = spec.builder(random_state, class_weight)
    except TypeError:
        # Some estimators (e.g. GaussianNB, KNN) ignore class_weight entirely
        model = spec.builder(random_state, None)

    start = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        raise ModelTrainingError(f"{spec.label} could not be trained: {exc}")
    training_time = time.time() - start

    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    metrics = compute_metrics(y_test, y_pred, y_proba, n_classes=n_classes)

    feature_importance = None
    if hasattr(model, "feature_importances_"):
        feature_importance = np.asarray(model.feature_importances_).tolist()
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        feature_importance = np.abs(coef).mean(axis=0).tolist() if coef.ndim > 1 else np.abs(coef).tolist()

    return {
        "model_key": model_key,
        "model_label": spec.label,
        "family": "ML",
        "training_time_sec": round(training_time, 3),
        "feature_importance": feature_importance,
        **metrics,
    }
