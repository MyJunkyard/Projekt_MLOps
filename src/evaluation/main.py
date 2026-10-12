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

from src.common.logsetup import setup_logging
from src.common.metrics import compute_metrics
from src.common.splits import get_split_masks
from src.config import load_config
from src.evaluation.plots import plot_actual_vs_predicted, residual_breakdown
from src.evaluation.reporting import (
    get_model_run_id,
    load_model_from_registry,
    load_test_features,
    log_evaluation_results,
    log_results_table,
    resolve_evaluate_columns,
)
from src.training.loader import drop_selected_nan_rows, open_superset

# Stable module name (not `__name__` — under `python -m` it is
# `"__main__"` and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.evaluation.main"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def main(run_id: str | None = None) -> None:
    """Orchestrate evaluation: metrics, plots, and residual breakdown.

    Args:
        run_id: Optional MLflow run ID to score on the test split instead of
            the champion alias (WS6 D10). When given, the model is loaded from
            that run, its columns come from that run's own schema artifact, and
            the metrics attach to that run — so a non-champion run (a baseline,
            or one arm of EXP-022) can be scored via ``make evaluate RUN_ID=``.
            ``None`` (the default) keeps every existing call site on the
            champion alias.
    """
    cfg = load_config()
    # Configure the *package* logger by explicit name (never `__name__` —
    # under `python -m` that resolves to "__main__" and would bypass the
    # configured handlers; see MODULE_LOGGER_NAME above). Package scope so
    # every sibling module logger (plots, reporting, ...) inherits the
    # handlers and level; leaf-scope would leave them unconfigured
    # (effective WARNING, INFO logs silently dropped).
    setup_logging(cfg.logging, logger_name="src.evaluation")
    reports_dir = Path("reports")

    logger.info("Stage: evaluation")
    logger.info("Opening feature superset")
    df, schema = open_superset(cfg)

    # WS6 (D10): resolve the scored run ONCE, then pass it to every consumer
    # so load, column resolution, and metric logging all agree on the same
    # run (no alias re-resolution between them). ``run_id`` given → score that
    # run; ``None`` → the champion alias.
    target_run = run_id if run_id is not None else get_model_run_id(cfg)
    logger.info(
        "Resolving feature columns for run %s",
        target_run if target_run is not None else "champion (alias)",
    )
    feature_cols, _ = resolve_evaluate_columns(cfg, schema, run_id=target_run)

    # One frame, one NaN policy. The superset keeps warm-up NaNs, so rows are
    # dropped only where the *resolved selection* (or the target) is NaN.
    # Everything below is derived from this single filtered frame, so
    # y_test/y_pred and df_test stay row-aligned for the residual breakdowns.
    logger.info("Building the test frame")
    df_clean = drop_selected_nan_rows(df, feature_cols, cfg.data.target_col)
    _, _, test_mask = get_split_masks(df_clean, cfg.data)
    df_test = df_clean[test_mask].reset_index(drop=True)

    X_test, y_test = load_test_features(df_test, feature_cols, cfg.data.target_col)

    logger.info("Loading model from registry")
    model = load_model_from_registry(cfg, run_id=target_run)

    logger.info("Computing predictions")
    y_pred = np.asarray(model.predict(X_test))

    logger.info("Computing metrics")
    metrics = compute_metrics(y_test, y_pred, cfg.evaluation.metrics)
    log_results_table(metrics)

    artifact_paths: list[str] = []

    # Generate plots if enabled
    if cfg.evaluation.generate_plots:
        reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating plots")
        plot_path = plot_actual_vs_predicted(
            y_test, y_pred, str(reports_dir / "actual_vs_predicted.png")
        )

        logger.info("Computing residual breakdown")
        breakdown_df = residual_breakdown(df_test, y_test, y_pred, cfg.evaluation)

        logger.debug("Residual breakdown:\n%s", breakdown_df.to_string(index=False))

        # Save residual breakdown to CSV
        breakdown_path = reports_dir / "residual_breakdown.csv"
        breakdown_df.to_csv(breakdown_path, index=False)
        logger.info("Residual breakdown saved to %s", breakdown_path)

        artifact_paths = [plot_path, str(breakdown_path)]

    # Log metrics and artifacts to MLflow, attached to the scored run (the
    # champion alias's run by default, or the explicit RUN_ID). Logging
    # outside an active run would silently create an orphan anonymous run.
    try:
        log_evaluation_results(cfg, metrics, artifact_paths, run_id=target_run)
    except Exception as e:
        logger.warning("Could not log results to MLflow: %s", e)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
