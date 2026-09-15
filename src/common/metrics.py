"""
common/metrics.py — Regression metric computation.

Moved verbatim from ``utils.py`` (Workstream 0 module restructure).
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    metrics_list: list[str],
) -> dict[str, float]:
    """Compute requested regression metrics and return as a dict.

    Args:
        y_true: Ground-truth target values. Array-like of shape
            (n_samples,); normalized to a 1-D float ndarray via
            ``np.asarray``.
        y_pred: Predicted target values. Array-like of shape
            (n_samples,); normalized the same way.
        metrics_list: Names of metrics to compute. Supported values:
            ``"rmse"``, ``"mae"``, ``"mape"``, ``"r2"``.

    Returns:
        A dict mapping each requested metric name to its float value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    results = {}
    for metric in metrics_list:
        if metric == "rmse":
            results["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif metric == "mae":
            results["mae"] = float(mean_absolute_error(y_true, y_pred))
        elif metric == "mape":
            # Avoid division by zero — mask zero prices
            mask = y_true != 0
            if mask.sum() > 0:
                results["mape"] = float(
                    np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                )
            else:
                results["mape"] = float("nan")
        elif metric == "r2":
            results["r2"] = float(r2_score(y_true, y_pred))
    return results
