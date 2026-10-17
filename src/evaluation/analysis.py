"""
evaluation/analysis.py — Shared residual-analysis entry point (WS4).

Composes the plot family in :mod:`src.evaluation.plots` with the extended
``residual_breakdown`` and ``src.common.metrics`` into a single call that turns
one fitted model + one split frame into the analysis artifact set. Two
call-sites use it: ``evaluation.main`` (test split) and ``training.main``
(val split, gated on ``evaluation.run_analysis_after_train``).

Imports nothing from ``src.training`` (decision **D4-a**): it takes an
already-masked split frame + explicit ``feature_cols``, so it never opens the
superset, resolves a selection, or touches the training package. The
import-guard test pins this so a future regression fails loudly.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)

import numpy as np
import pandas as pd

from src.common.metrics import compute_metrics
from src.config.models import PipelineConfig
from src.evaluation.plots import (
    plot_actual_vs_predicted,
    plot_actual_vs_predicted_window,
    plot_residual_breakdown,
    plot_residuals_over_time,
    residual_breakdown,
)

MODULE_LOGGER_NAME = "src.evaluation.analysis"
logger = logging.getLogger(MODULE_LOGGER_NAME)


@dataclass(frozen=True)
class AnalysisResult:
    """Metrics + written artifact paths for one analysed split (D4-b).

    Attributes:
        metrics: Metric name -> value from ``compute_metrics`` over
            ``cfg.evaluation.metrics`` (single source of truth for the
            caller's results table *and* MLflow logging).
        artifact_paths: Paths of the files actually written under
            ``output_dir``, split-prefixed (D4-c). Empty when
            ``evaluation.generate_plots`` is false.
    """

    metrics: dict[str, float]
    artifact_paths: tuple[str, ...]


def run_analysis(
    cfg: PipelineConfig,
    model,  # has .predict; raw estimator or pyfunc
    df: pd.DataFrame,  # ONE split, NaN-filtered, row-aligned
    split: str,  # "val" | "test" — drives the filename prefix
    feature_cols: list[str],  # the model's input columns (D4-b)
    output_dir: str | Path = "reports",
) -> AnalysisResult:
    """Compute metrics + all residual plots + breakdown CSVs for one split.

    The full artifact set for a split (written only when
    ``cfg.evaluation.generate_plots`` is true):

    - ``{split}_residuals_over_time.png`` — residual series + rolling band;
    - ``{split}_actual_vs_predicted.png`` — full-split overlay;
    - ``{split}_residual_breakdown.csv`` + ``{split}_residual_breakdown.png``
      — grouped mean-absolute-residual table (categoricals + price quantiles
      + direction) and its faceted chart;
    - ``{split}_actual_vs_predicted_window.png`` — only when
      ``cfg.evaluation.sample_week`` falls inside this split (D4-d; the
      other split logs a WARNING and emits none).

    Args:
        cfg: Validated pipeline config (target column, metrics, plot
            switches, breakdown columns, sample week).
        model: Anything with ``.predict(X)`` — a raw estimator (train-time)
            or an MLflow pyfunc model (evaluate-time).
        df: ONE split of the cleaned superset, NaN-filtered and row-aligned
            with the model's inputs; must carry the target column, the
            timestamp column, and any ``evaluation.residual_breakdown``
            grouping columns that exist.
        split: ``"val"`` or ``"test"`` — the artifact filename prefix that
            lets the two call-sites coexist in ``reports/`` (D4-c).
        feature_cols: The model's input columns, in canonical order.
        output_dir: Where artifacts are written; defaults to ``"reports"``
            (tests pass their own dir to stay hermetic).

    Returns:
        The metrics dict + the exact paths written (never a bare list —
        D4-b).

    Guard rails:
        - ``generate_plots: false`` still computes and returns metrics but
          writes no PNG and no CSV (the hermiticity contract), so
          ``artifact_paths`` is empty.
        - Empty/one-row frames never raise in the plot functions.
        - ``sample_week`` outside this split → WARNING, no window artifact.
    """
    target_col = cfg.data.target_col
    y_true = np.asarray(df[target_col], dtype=float)
    X = df[feature_cols]
    y_pred = np.asarray(model.predict(X), dtype=float)

    metrics = compute_metrics(y_true, y_pred, cfg.evaluation.metrics)

    # Plots off = no `reports/` writes at all (existing hermiticity contract
    # the tests rely on); metrics are still returned so evaluate/train can
    # log them regardless of the plot switch.
    if not cfg.evaluation.generate_plots:
        logger.info(
            "generate_plots is false; metrics only for the %s split", split
        )
        return AnalysisResult(metrics=metrics, artifact_paths=())


    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    date_col = cfg.data.date_col

    logger.info("Generating %s-split residual plots", split)
    ts: pd.Series | None = None
    if date_col in df.columns:
        ts = pd.to_datetime(df[date_col])
        residuals = y_true - y_pred
        paths.append(
            plot_residuals_over_time(
                ts, residuals, str(out / f"{split}_residuals_over_time.png")
            )
        )
    else:
        # Defensive: a frame without timestamps can still produce the
        # index-based plots; only the time-axis ones are skipped.
        logger.warning(
            "Column %r missing from the %s-split frame; skipping time plots",
            date_col,
            split,
        )

    paths.append(
        plot_actual_vs_predicted(
            y_true, y_pred, str(out / f"{split}_actual_vs_predicted.png")
        )
    )

    logger.info("Computing residual breakdown")
    breakdown_df = residual_breakdown(df, y_true, y_pred, cfg.evaluation)
    logger.debug("Residual breakdown:\n%s", breakdown_df.to_string(index=False))

    breakdown_path = out / f"{split}_residual_breakdown.csv"
    breakdown_df.to_csv(breakdown_path, index=False)
    logger.info("Residual breakdown saved to %s", breakdown_path)
    paths.append(str(breakdown_path))
    paths.append(
        plot_residual_breakdown(
            breakdown_df, str(out / f"{split}_residual_breakdown.png")
        )
    )

    # D4-d single owner: the sample-week window is emitted only by the split
    # that actually contains it. The other split logs a WARNING and writes
    # no window artifact.
    sample_week = cfg.evaluation.sample_week
    if sample_week is not None and ts is not None:
        start_ts = pd.Timestamp(sample_week)
        end_ts = start_ts + pd.Timedelta(days=7)
        if start_ts.tz is None and ts.dt.tz is not None:
            start_ts = start_ts.tz_localize(ts.dt.tz)
            end_ts = end_ts.tz_localize(ts.dt.tz)
        if bool(((ts >= start_ts) & (ts < end_ts)).any()):
            window_path = plot_actual_vs_predicted_window(
                ts,
                y_true,
                y_pred,
                start_ts,
                end_ts,
                str(out / f"{split}_actual_vs_predicted_window.png"),
            )
            if window_path is not None:
                paths.append(window_path)
        else:
            logger.warning(
                "Sample week %s is not inside the %s split; skipping "
                "actual-vs-predicted window plot (single owner, D4-d)",
                sample_week,
                split,
            )
    elif sample_week is not None:
        logger.warning(
            "Column %r missing from the %s-split frame; cannot emit the "
            "sample-week window for %s",
            date_col,
            split,
            sample_week,
        )

    return AnalysisResult(metrics=metrics, artifact_paths=tuple(paths))


__all__ = ["AnalysisResult", "run_analysis"]

