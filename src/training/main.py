"""
training/main.py — Training orchestration.

Orchestrates: load features → fit XGBoost + baselines → evaluate on
val → log to MLflow. The only place in the training package that knows
the run order. Moved verbatim from ``train.py`` (Workstream 0 module
restructure).
"""

import logging

import numpy as np

from src.common.logsetup import setup_logging
from src.common.metrics import compute_metrics
from src.config import load_config
from src.training.baselines import (
    train_baseline_persistence,
    train_baseline_seasonal_naive,
)
from src.training.loader import get_feature_names, load_features, load_model
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
    processed_path = cfg.data.processed_path

    logger.info("Stage: training")
    logger.info("Loading features")
    X_train, y_train, X_val, y_val, X_test, y_test = load_features(
        processed_path, cfg.data
    )
    feature_names = get_feature_names(processed_path, cfg.data)

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
    # Primary model: promoted to the champion alias
    log_to_mlflow(
        model,
        metrics,
        cfg,
        feature_names=feature_names,
        run_name="xgboost-v1",
        promote_to_production=True,
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
