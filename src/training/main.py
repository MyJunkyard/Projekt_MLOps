"""
training/main.py — Training orchestration.

Orchestrates: load features → fit XGBoost + baselines → evaluate on
val → log to MLflow → val-split residual analysis (WS4, gated on
``evaluation.run_analysis_after_train``). The only place in the
training package that knows the run order. Moved verbatim from
``train.py`` (Workstream 0 module restructure).
"""

import logging

import numpy as np

from src.common.logsetup import setup_logging
from src.common.metrics import compute_metrics
from src.common.splits import get_split_masks
from src.config import load_config
from src.evaluation.analysis import run_analysis
from src.evaluation.reporting import log_evaluation_results
from src.training.baselines import (
    train_baseline_persistence,
    train_baseline_seasonal_naive,
)
from src.training.loader import load_model, load_superset_features
from src.training.registry import log_to_mlflow

# Stable module logger name (not __name__ — `python -m` sets __name__ to
# "__main__", which would bypass the configured src handlers).
MODULE_LOGGER_NAME = "src.training.main"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def main():
    """Orchestrate training pipeline: train XGBoost + baselines, log to MLflow."""
    cfg = load_config()
    # Configure the *package* logger by explicit name (never `__name__` —
    # under `python -m` that resolves to "__main__" and would bypass the
    # configured handlers; see MODULE_LOGGER_NAME above). Package scope so
    # every sibling module logger (loader, registry, baselines, ...)
    # inherits the handlers and level; leaf-scope would leave them
    # unconfigured (effective WARNING, INFO logs silently dropped).
    setup_logging(cfg.logging, logger_name="src.training")
    logger.info("Stage: training")
    logger.info("Loading features from the superset")
    loaded = load_superset_features(cfg)
    X_splits, y_splits, feature_names, subset_schema, clean_frame = loaded
    X_train, y_train = X_splits[0], y_splits[0]
    X_val, y_val = X_splits[1], y_splits[1]

    # Per-split row counts (WS6 D6): logged on every run so a silent row-set
    # shift is visible in the compare table. len() counts rows *after* the
    # warm-up/NaN policy (drop_selected_nan_rows in common.selection) — the
    # number the model actually saw, the only one worth comparing.
    row_counts = {
        name: int(len(split))
        for name, split in zip(("train", "val", "test"), X_splits, strict=True)
    }

    # --- Train XGBoost ---
    logger.info("Loading model: %s", cfg.model.type)
    model = load_model(cfg.model)
    logger.debug("Model: %s", model)

    logger.info("Training XGBoost")
    model.fit(X_train, y_train)
    logger.info("Training complete")

    logger.info("Evaluating on validation set")
    y_pred = np.asarray(model.predict(X_val))
    metrics = compute_metrics(y_val, y_pred, cfg.evaluation.metrics)
    for name, value in metrics.items():
        logger.info("Validation %s: %.4f", name, value)

    logger.info("Logging XGBoost to MLflow")
    # Primary model: promoted to the champion alias. The *subset* schema is
    # logged so evaluation and serving read the model's own input contract
    # (WS0 decision D-c) rather than re-deriving it from config. The returned
    # run id anchors the WS4 val-split analysis artifacts below.
    primary_run_id = log_to_mlflow(
        model,
        metrics,
        cfg,
        feature_names=feature_names,
        feature_schema=subset_schema,
        run_name="xgboost-v1",
        promote_to_production=True,
        row_counts=row_counts,
    )

    # WS4 (D4): val-split residual analysis, gated on the config flag. The
    # artifacts attach to the PRIMARY run — the one carrying the champion
    # alias — so train (val) and evaluate (test) emit the same artifact
    # family under split-prefixed names (D4-c). The empty metrics dict means
    # "artifacts only": train already logged its val metrics above. A failed
    # attach is a WARNING, never a failed training run (mirrors evaluate).
    if cfg.evaluation.run_analysis_after_train:
        _, val_mask, _ = get_split_masks(clean_frame, cfg.data)
        df_val = clean_frame[val_mask].reset_index(drop=True)
        result = run_analysis(cfg, model, df_val, "val", feature_names)
        try:
            log_evaluation_results(
                cfg, {}, list(result.artifact_paths), run_id=primary_run_id
            )
        except Exception as exc:
            logger.warning(
                "Could not attach val-split analysis artifacts: %s", exc
            )
        logger.info(
            "Val-split residual analysis attached to run %s", primary_run_id
        )

    # --- Train baselines ---
    # Baselines are registered for comparison but NEVER promoted
    # (promote_to_production defaults to False).
    logger.info("Training baseline: persistence")
    persistence_model, persistence_metrics = train_baseline_persistence(
        X_train, y_train, X_val, y_val
    )
    for name, value in persistence_metrics.items():
        logger.info("Persistence baseline validation %s: %.4f", name, value)

    logger.info("Logging persistence baseline to MLflow")
    log_to_mlflow(
        persistence_model,
        persistence_metrics,
        cfg,
        run_name="baseline-persistence",
        row_counts=row_counts,
    )

    logger.info("Training baseline: seasonal naive")
    seasonal_model, seasonal_metrics = train_baseline_seasonal_naive(
        X_train, y_train, X_val, y_val
    )
    for name, value in seasonal_metrics.items():
        logger.info("Seasonal naive baseline validation %s: %.4f", name, value)

    logger.info("Logging seasonal naive baseline to MLflow")
    log_to_mlflow(
        seasonal_model,
        seasonal_metrics,
        cfg,
        run_name="baseline-seasonal-naive",
        row_counts=row_counts,
    )

    # --- Summary ---
    logger.info(
        "Model comparison (validation RMSE): xgboost=%.4f, "
        "persistence=%.4f, seasonal_naive=%.4f",
        metrics["rmse"],
        persistence_metrics["rmse"],
        seasonal_metrics["rmse"],
    )

    logger.info("Training complete")


if __name__ == "__main__":
    main()
