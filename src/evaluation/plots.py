"""
evaluation/plots.py — Plot generation for evaluation reporting.

Moved verbatim from ``evaluate.py`` (Workstream 0 module restructure).
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULE_LOGGER_NAME = "src.evaluation.plots"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def plot_actual_vs_predicted(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    output_path: str,
) -> str:
    """Generate an actual-vs-predicted line plot and save as PNG.

    Args:
        y_true: Ground-truth target values. Array-like of shape
            (n_samples,); normalized to a 1-D float ndarray via
            ``np.asarray``.
        y_pred: Predicted target values. Array-like of the same length
            as ``y_true``; normalized the same way.
        output_path: Path to save the PNG file.

    Returns:
        The path the plot was saved to.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_true, label="Actual", alpha=0.8, linewidth=1.5)
    ax.plot(y_pred, label="Predicted", alpha=0.8, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title("Actual vs Predicted — Test Set")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path_obj, dpi=150)
    plt.close(fig)

    logger.info("Plot saved to %s", output_path_obj)
    return str(output_path_obj)


def residual_breakdown(
    df: pd.DataFrame,
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    cfg: dict,
) -> pd.DataFrame:
    """Compute mean absolute residual grouped by categorical variables.

    For each categorical variable in ``evaluation.residual_breakdown``
    (e.g. hour, day_of_week, month, is_holiday), this computes the mean
    absolute residual for each category value.

    Args:
        df: The original (test-split) DataFrame with the categorical
            grouping columns; must be row-aligned with ``y_true`` /
            ``y_pred`` (same order and length).
        y_true: Ground-truth target values. Array-like of shape
            (n_samples,); normalized to a 1-D float ndarray via
            ``np.asarray``.
        y_pred: Predicted target values. Array-like of the same length
            as ``y_true``; normalized the same way.
        cfg: Configuration dict with ``evaluation.residual_breakdown``
            (list of grouping column names).

    Returns:
        A DataFrame with columns ``grouping`` (str), ``category``,
        ``mean_abs_residual`` (float), and ``count`` (int).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = np.abs(y_true - y_pred)
    breakdown_cols = cfg["evaluation"]["residual_breakdown"]

    rows = []
    for col in breakdown_cols:
        if col not in df.columns:
            continue
        for category in sorted(df[col].unique()):
            mask = (df[col] == category).to_numpy()
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "grouping": col,
                    "category": category,
                    "mean_abs_residual": float(residuals[mask].mean()),
                    "count": int(mask.sum()),
                }
            )

    return pd.DataFrame(rows)
