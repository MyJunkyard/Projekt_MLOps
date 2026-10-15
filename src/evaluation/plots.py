"""
evaluation/plots.py — Plot generation for evaluation reporting.

Moved verbatim from ``evaluate.py`` (Workstream 0 module restructure).

WS4 (stage 4) adds the residual-analysis plot family —
``plot_residuals_over_time``, ``plot_actual_vs_predicted_window`` and
``plot_residual_breakdown`` — alongside the existing
``plot_actual_vs_predicted``. Every function keeps the same shape: arrays /
timestamps + output path in, saved path out; no config, no MLflow, no superset.
Only ``src/evaluation/analysis.py`` composes them.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.models import EvaluationConfig

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
    evaluation: EvaluationConfig,
    quantiles: int | None = None,
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
        evaluation: ``EvaluationConfig`` with ``residual_breakdown``
            (list of grouping column names) and ``residual_quantiles``
            (default price-quantile bin count when ``quantiles`` is ``None``).
        quantiles: Number of price-quantile bins for an extra ``price_quantile``
            grouping (WS4). ``None`` (default) uses
            ``evaluation.residual_quantiles``; ``0`` disables the quantile
            grouping. Ties/duplicate bin edges are dropped (``duplicates="drop"``).

    Returns:
        A DataFrame with columns ``grouping`` (str), ``category``,
        ``mean_abs_residual`` (float), and ``count`` (int). ``grouping`` is one
        of the categorical columns, ``"price_quantile"`` (when enabled), or
        ``"direction"`` (always present — the price-change direction of the
        *actual*, first row excluded).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = np.abs(y_true - y_pred)
    breakdown_cols = evaluation.residual_breakdown

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

    # Price-quantile grouping (WS4): bin the ACTUAL price with qcut and group
    # the residuals by bin. ``quantiles=None`` defers to the config default;
    # ``0`` disables. Duplicate bin edges (heavy ties) are dropped; a fully
    # degenerate series still raises inside qcut, so guard it as a no-op.
    n_quantiles = (
        evaluation.residual_quantiles if quantiles is None else quantiles
    )
    if n_quantiles and n_quantiles > 0:
        try:
            bins = pd.qcut(y_true, q=n_quantiles, duplicates="drop")
            for category in bins.categories:
                mask = np.asarray(bins == category)
                if mask.sum() == 0:
                    continue
                rows.append(
                    {
                        "grouping": "price_quantile",
                        "category": str(category),
                        "mean_abs_residual": float(residuals[mask].mean()),
                        "count": int(mask.sum()),
                    }
                )
        except ValueError as exc:
            logger.warning(
                "Could not build the price-quantile breakdown (%s); skipping it",
                exc,
            )

    # Price-change direction grouping (WS4, Series B): sign of the actual's
    # period-over-period change, computed WITHIN the split so the first row
    # (NaN) is excluded and no leakage crosses the split boundary (D4-f).
    direction = np.sign(pd.Series(y_true) - pd.Series(y_true).shift(1))
    for sign_value, label in ((1.0, "rising"), (-1.0, "falling"), (0.0, "flat")):
        mask = (direction == sign_value).to_numpy()
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "grouping": "direction",
                "category": label,
                "mean_abs_residual": float(residuals[mask].mean()),
                "count": int(mask.sum()),
            }
        )

    return pd.DataFrame(rows)


def plot_residuals_over_time(
    timestamps: pd.Series | np.ndarray,
    residuals: np.ndarray | list[float],
    output_path: str,
) -> str:
    """Plot the residual series over time with a rolling-mean band.

    Surfaces drift and seasonality in the errors that a single scalar metric
    cannot show. The raw residuals are drawn as a faint line; a centered
    rolling mean (solid) with a ±1σ band (shaded) makes slow bias and
    heteroscedasticity visible.

    Args:
        timestamps: Datetime values (tz-aware UTC, from the superset
            ``timestamp`` column) aligned with ``residuals``.
        residuals: Signed residuals (``y_true - y_pred``), array-like of
            shape (n_samples,); normalized to a 1-D float ndarray.
        output_path: Path to save the PNG file.

    Returns:
        The path the plot was saved to.
    """
    residuals = np.asarray(residuals, dtype=float)
    ts = pd.Series(pd.to_datetime(pd.Series(np.asarray(timestamps))))
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        ts,
        residuals,
        color="tab:gray",
        linewidth=0.7,
        alpha=0.6,
        label="residual",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)

    # Rolling mean ± 1σ. A short series (or a single row) must not raise, so
    # the window is clamped to the data length and min_periods=1.
    n = len(residuals)
    window = min(max(n // 10, 1), 50)
    resid_series = pd.Series(residuals)
    roll = resid_series.rolling(window, min_periods=1)
    mean = roll.mean().to_numpy()
    std = roll.std(ddof=0).to_numpy()
    ax.plot(ts, mean, color="tab:blue", linewidth=1.8, label="rolling mean")
    ax.fill_between(
        ts.to_numpy(),
        mean - std,
        mean + std,
        color="tab:blue",
        alpha=0.15,
        label="±1σ",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Residual (EUR/MWh)")
    ax.set_title("Residuals over time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path_obj, dpi=150)
    plt.close(fig)

    logger.info("Plot saved to %s", output_path_obj)
    return str(output_path_obj)


def plot_actual_vs_predicted_window(
    timestamps: pd.Series | np.ndarray,
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    output_path: str,
) -> str | None:
    """Plot actual-vs-predicted for a single sample window ``[start, end)``.

    A week-scoped companion to :func:`plot_actual_vs_predicted`: where the full
    plot is unreadable over thousands of hours, this zooms into one window
    (e.g. the configured ``evaluation.sample_week``) so short-horizon skill and
    spike capture are visible.

    Args:
        timestamps: Datetime values aligned with ``y_true`` / ``y_pred``.
        y_true: Ground-truth target values, array-like of shape (n_samples,).
        y_pred: Predicted values, same length as ``y_true``.
        start: Inclusive window start (ISO string or Timestamp).
        end: Exclusive window end (ISO string or Timestamp).
        output_path: Path to save the PNG file.

    Returns:
        The saved path, or ``None`` (with a WARNING) when the window selects
        no rows — an out-of-range or mis-typed window is a no-op, not an error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ts = pd.Series(pd.to_datetime(pd.Series(np.asarray(timestamps))))

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tz is None and ts.dt.tz is not None:
        start_ts = start_ts.tz_localize(ts.dt.tz)
    if end_ts.tz is None and ts.dt.tz is not None:
        end_ts = end_ts.tz_localize(ts.dt.tz)

    mask = ((ts >= start_ts) & (ts < end_ts)).to_numpy()
    if not mask.any():
        logger.warning(
            "Sample window [%s, %s) selects no rows; skipping plot %s",
            start,
            end,
            output_path,
        )
        return None

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts[mask], y_true[mask], label="Actual", linewidth=1.6)
    ax.plot(
        ts[mask],
        y_pred[mask],
        label="Predicted",
        linewidth=1.6,
        linestyle="--",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title(f"Actual vs Predicted — {start} → {end}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path_obj, dpi=150)
    plt.close(fig)

    logger.info("Plot saved to %s", output_path_obj)
    return str(output_path_obj)



def plot_residual_breakdown(
    breakdown_df: pd.DataFrame, output_path: str
) -> str:
    """Render a residual-breakdown table as one bar chart per grouping column.

    The long-format frame (``grouping``, ``category``, ``mean_abs_residual``,
    ``count``) is faceted: one subplot per distinct ``grouping`` value (hour,
    day_of_week, …, price_quantile, direction), each a bar chart of mean
    absolute residual by category. One row of a single grouping still renders.

    Args:
        breakdown_df: The frame returned by ``residual_breakdown`` (may be
            empty, which yields an empty-but-valid figure).
        output_path: Path to save the PNG file.

    Returns:
        The path the plot was saved to.
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    groupings = (
        list(dict.fromkeys(breakdown_df["grouping"]))
        if not breakdown_df.empty
        else []
    )
    n_axes = max(len(groupings), 1)

    fig, axes = plt.subplots(
        n_axes, 1, figsize=(10, 3.2 * n_axes), squeeze=False
    )
    axes_flat = axes[:, 0]

    if not groupings:
        axes_flat[0].axis("off")
        axes_flat[0].text(
            0.5, 0.5, "No residual-breakdown rows", ha="center", va="center"
        )
    else:
        for ax, grouping in zip(axes_flat, groupings, strict=True):
            sub = breakdown_df[breakdown_df["grouping"] == grouping]
            ax.bar(
                sub["category"].astype(str),
                sub["mean_abs_residual"],
                color="tab:blue",
                alpha=0.8,
            )
            ax.set_title(str(grouping))
            ax.set_ylabel("Mean |residual|")
            ax.grid(True, axis="y", alpha=0.3)
            if len(sub) > 6:
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha("right")

    fig.tight_layout()
    fig.savefig(output_path_obj, dpi=150)
    plt.close(fig)

    logger.info("Plot saved to %s", output_path_obj)
    return str(output_path_obj)

