"""
evaluation/main.py — Evaluation orchestration.

Orchestrates: load test features → predict with champion model →
compute metrics → plots/residual breakdown → log to MLflow. The only
place in the evaluation package that knows the run order. Moved
verbatim from ``evaluate.py`` (Workstream 0 module restructure).
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)

import numpy as np
import pandas as pd

from src.common.logsetup import setup_logging
from src.common.metrics import compute_metrics
from src.common.splits import get_split_masks
from src.config import load_config
from src.evaluation.plots import plot_actual_vs_predicted, residual_breakdown
from src.evaluation.reporting import (
    load_model_from_registry,
    load_test_features,
    log_evaluation_results,
    log_results_table,
)

# Stable module name (not `__name__` — under `python -m` it is
# `"__main__"` and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.evaluation.main"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def main():
    """Orchestrate evaluation: metrics, plots, and residual breakdown."""
    cfg = load_config()
    # Configure the *package* logger by explicit name (never `__name__` —
    # under `python -m` that resolves to "__main__" and would bypass the
    # configured handlers; see MODULE_LOGGER_NAME above). Package scope so
    # every sibling module logger (plots, reporting, ...) inherits the
    # handlers and level; leaf-scope would leave them unconfigured
    # (effective WARNING, INFO logs silently dropped).
    setup_logging(cfg, logger_name="src.evaluation")
    processed_path = cfg["data"]["processed_path"]
    reports_dir = Path("reports")

    logger.info("Stage: evaluation")
    logger.info("Loading test features")
    df = pd.read_parquet(processed_path)
    X_test, y_test = load_test_features(processed_path, cfg)

    # Get test DataFrame columns for residual breakdown
    _, _, test_mask = get_split_masks(df, cfg)
    df_test = df.iloc[test_mask].reset_index(drop=True)

    logger.info("Loading model from registry")
    model = load_model_from_registry(cfg)

    logger.info("Computing predictions")
    y_pred = np.asarray(model.predict(X_test))

    logger.info("Computing metrics")
    metrics = compute_metrics(y_test, y_pred, cfg["evaluation"]["metrics"])
    log_results_table(metrics)

    artifact_paths: list[str] = []

    # Generate plots if enabled
    if cfg["evaluation"].get("generate_plots", False):
        reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating plots")
        plot_path = plot_actual_vs_predicted(
            y_test, y_pred, str(reports_dir / "actual_vs_predicted.png")
        )

        logger.info("Computing residual breakdown")
        breakdown_df = residual_breakdown(df_test, y_test, y_pred, cfg)

        logger.debug("Residual breakdown:\n%s", breakdown_df.to_string(index=False))

        # Save residual breakdown to CSV
        breakdown_path = reports_dir / "residual_breakdown.csv"
        breakdown_df.to_csv(breakdown_path, index=False)
        logger.info("Residual breakdown saved to %s", breakdown_path)

        artifact_paths = [plot_path, str(breakdown_path)]

    # Log metrics and artifacts to MLflow, attached to the training run
    # that produced the model (logging outside an active run would
    # silently create an orphan anonymous run)
    try:
        log_evaluation_results(cfg, metrics, artifact_paths)
    except Exception as e:
        logger.warning("Could not log results to MLflow: %s", e)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
