"""
Applies the selected imbalance-handling technique to the TRAINING fold only.
This module must never see the test fold — that guarantee is enforced by the
orchestrator, which only ever passes X_train / y_train in here.
"""
import logging

import numpy as np

from app.utils.errors import exception_location

logger = logging.getLogger("uvicorn.error")


class ImbalanceError(Exception):
    pass


def apply_imbalance_technique(X_train, y_train, method: str, random_state: int = 42):
    """
    Returns (X_resampled, y_resampled, class_weight_dict_or_none).

    `method == "class_weight"` does not resample; instead it returns a
    class_weight dict for models that support it natively.
    """
    if method in ("none", None):
        return X_train, y_train, None

    if method == "class_weight":
        classes, counts = np.unique(y_train, return_counts=True)
        total = counts.sum()
        weights = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
        return X_train, y_train, weights

    # Force clean dtypes: samplers/nearest-neighbour code expects float64 features and
    # integer class labels, and mismatches raise "Cannot cast array data" TypeErrors.
    X_train = np.ascontiguousarray(np.asarray(X_train), dtype=np.float64)
    y_train = np.asarray(y_train).astype(np.int64)

    # Resampling techniques need at least a few minority samples to work with.
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min())

    try:
        if method == "random_oversample":
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=random_state)
        elif method == "random_undersample":
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=random_state)
        elif method == "smote":
            from imblearn.over_sampling import SMOTE
            k_neighbors = max(1, min(5, min_count - 1))
            sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        elif method == "adasyn":
            from imblearn.over_sampling import ADASYN
            k_neighbors = max(1, min(5, min_count - 1))
            sampler = ADASYN(random_state=random_state, n_neighbors=k_neighbors)
        elif method == "smoteenn":
            from imblearn.combine import SMOTEENN
            sampler = SMOTEENN(random_state=random_state)
        elif method == "smotetomek":
            from imblearn.combine import SMOTETomek
            sampler = SMOTETomek(random_state=random_state)
        else:
            raise ImbalanceError(f"Unknown imbalance technique: {method}")

        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        return X_resampled, np.asarray(y_resampled).astype(np.int64), None

    except ImbalanceError:
        raise
    except ValueError as exc:
        raise ImbalanceError(
            f"Could not apply '{method}': the minority class may have too few "
            f"samples ({min_count}) for this technique. Details: {exc}"
        )
    except Exception as exc:
        logger.exception("Resampling with '%s' failed", method)
        raise ImbalanceError(
            f"Could not apply '{method}' ({type(exc).__name__}: {' '.join(str(exc).split())[:200]}) "
            f"[at {exception_location(exc)}]."
        )