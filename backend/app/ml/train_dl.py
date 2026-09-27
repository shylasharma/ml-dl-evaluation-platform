import time
from typing import Optional, Dict, Any
import numpy as np

from app.ml.metrics import compute_metrics
from app.ml.registry import DL_MODELS
from app.ml.train_ml import ModelTrainingError


def train_and_evaluate_dl(
    model_key: str,
    X_train, y_train, X_test, y_test,
    n_classes: int,
    class_weight: Optional[dict] = None,
    epochs: int = 30,
    batch_size: int = 32,
    random_state: int = 42,
) -> Dict[str, Any]:
    if model_key not in DL_MODELS:
        raise ModelTrainingError(f"Unknown DL model '{model_key}'.")

    spec = DL_MODELS[model_key]

    try:
        import tensorflow as tf
        tf.random.set_seed(random_state)
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        raise ModelTrainingError(
            f"{spec.label} could not be trained because TensorFlow is not installed. "
            f"Run `pip install tensorflow` in the backend environment."
        )

    input_dim = X_train.shape[1]

    try:
        model = spec.builder(input_dim, n_classes)
    except Exception as exc:
        raise ModelTrainingError(f"{spec.label} could not be built: {exc}")

    y_train_arr = y_train.astype("float32") if n_classes <= 2 else y_train.astype("int32")
    y_test_arr = y_test.astype("float32") if n_classes <= 2 else y_test.astype("int32")

    start = time.time()
    try:
        history = model.fit(
            X_train, y_train_arr,
            validation_split=0.15,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0,
        )
    except Exception as exc:
        raise ModelTrainingError(f"{spec.label} training failed: {exc}")
    training_time = time.time() - start

    raw_pred = model.predict(X_test, verbose=0)

    if n_classes <= 2:
        proba_pos = raw_pred.reshape(-1)
        y_proba = np.stack([1 - proba_pos, proba_pos], axis=1)
        y_pred = (proba_pos >= 0.5).astype(int)
    else:
        y_proba = raw_pred
        y_pred = np.argmax(raw_pred, axis=1)

    metrics = compute_metrics(y_test, y_pred, y_proba, n_classes=n_classes)

    training_history = {
        "loss": [float(v) for v in history.history.get("loss", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "accuracy": [float(v) for v in history.history.get("accuracy", [])],
        "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
    }

    return {
        "model_key": model_key,
        "model_label": spec.label,
        "family": "DL",
        "training_time_sec": round(training_time, 3),
        "epochs_run": len(training_history["loss"]),
        "training_history": training_history,
        "feature_importance": None,
        **metrics,
    }
