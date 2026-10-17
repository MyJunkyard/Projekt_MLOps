"""
evaluation/main.py — Evaluation orchestration.

Orchestrates: open the superset → resolve the scored run + feature
columns → build the test frame → run the shared residual analysis
(metrics + plots + breakdown, WS4) → log to MLflow. The only place in
the evaluation package that knows the run order. Moved verbatim from
``evaluate.py`` (Workstream 0 module restructure).
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)

from src.common.logsetup import setup_logging
from src.common.selection import drop_selected_nan_rows, open_superset
from src.common.splits import get_split_masks
from src.config import load_config
from src.evaluation.analysis import run_analysis
from src.evaluation.reporting import (
    get_model_run_id,
    load_model_from_registry,
    log_evaluation_results,
    log_results_table,
    resolve_evaluate_columns,
)

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
    # Everything below derives from this single filtered frame, so the
    # analysis sees exactly the rows the split owns (WS0 row-alignment).
    logger.info("Building the test frame")
    df_clean = drop_selected_nan_rows(df, feature_cols, cfg.data.target_col)
    _, _, test_mask = get_split_masks(df_clean, cfg.data)
    df_test = df_clean[test_mask].reset_index(drop=True)

    logger.info("Loading model from registry")
    model = load_model_from_registry(cfg, run_id=target_run)

    # WS4: one call owns metrics + every residual artifact for the split
    # (split-prefixed, D4-c). `evaluate` always analyses — it is the scoring
    # stage; `run_analysis_after_train` governs only `train`. The results
    # table reads `result.metrics` so there is a single source of truth;
    # `generate_plots: false` keeps the hermetic path (metrics only, no
    # `reports/` writes, empty `artifact_paths`).
    logger.info("Running test-split residual analysis")
    result = run_analysis(
        cfg, model, df_test, "test", feature_cols, output_dir=reports_dir
    )
    log_results_table(result.metrics)

    # Log metrics and artifacts to MLflow, attached to the scored run (the
    # champion alias's run by default, or the explicit RUN_ID). Logging
    # outside an active run would silently create an orphan anonymous run.
    try:
        log_evaluation_results(
            cfg, result.metrics, list(result.artifact_paths), run_id=target_run
        )
    except Exception as e:
        logger.warning("Could not log results to MLflow: %s", e)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
