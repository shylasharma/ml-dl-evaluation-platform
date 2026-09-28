"""Make experiment results safe to store and send as strict JSON."""
import math
import numpy as np


def sanitize_for_json(obj):
    """Recursively convert numpy types to plain Python and NaN/+-inf to None."""
    if isinstance(obj, dict):
        return {
            (k if (k is None or isinstance(k, (str, int, float, bool))) else str(k)): sanitize_for_json(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, np.generic):
        return sanitize_for_json(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj
