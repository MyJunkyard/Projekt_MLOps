"""
utils.py — Shared helpers for pipeline modules.

Common utilities used across the pipeline (config loading, metric
computation, train/val/test splitting) live here to avoid duplication
and keep a single source of truth.
"""

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_config(config_path: str = "params.yaml") -> dict:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary of configuration values.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, metrics_list: list[str]
) -> dict[str, float]:
    """Compute requested regression metrics and return as a dict.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.
        metrics_list: Names of metrics to compute. Supported values:
            ``"rmse"``, ``"mae"``, ``"mape"``, ``"r2"``.

    Returns:
        A dict mapping each requested metric name to its float value.
    """
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


def get_split_masks(
    df: pd.DataFrame, cfg: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for train/val/test splits based on config dates.

    Args:
        df: DataFrame with a tz-aware ``timestamp`` column.
        cfg: Configuration dict with ``data.train_end`` and ``data.val_end``
            date boundaries.

    Returns:
        A tuple ``(train_mask, val_mask, test_mask)`` of boolean numpy arrays.
    """
    # Boundaries are tz-naive in params.yaml; data is tz-aware (UTC)
    train_end = pd.Timestamp(cfg["data"]["train_end"], tz="UTC")
    val_end = pd.Timestamp(cfg["data"]["val_end"], tz="UTC")

    train_mask = (df["timestamp"] < train_end).to_numpy()
    val_mask = ((df["timestamp"] >= train_end) & (df["timestamp"] < val_end)).to_numpy()
    test_mask = (df["timestamp"] >= val_end).to_numpy()
    return train_mask, val_mask, test_mask