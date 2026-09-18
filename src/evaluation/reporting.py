"""
evaluation/reporting.py — MLflow result reporting and test-feature loading.

Moved verbatim from ``evaluate.py`` (Workstream 0 module restructure).
"""

import logging

import mlflow
import numpy as np
import pandas as pd

from src.common.splits import get_split_masks
from src.config.models import DataConfig, PipelineConfig

MODULE_LOGGER_NAME = "src.evaluation.reporting"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def load_model_from_registry(cfg: PipelineConfig):
    """Load model from MLflow model registry.

    Uses the model alias specified in params.yaml (default: champion).
    An alias points to exactly one registered version, so the loaded model
    is unambiguous.

    Args:
        cfg: The validated ``PipelineConfig`` (uses ``mlflow`` and
            ``serving`` sections).

    Returns:
        A loaded MLflow pyfunc model.
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    alias = cfg.serving.model_alias
    model_uri = f"models:/{cfg.mlflow.model_name}@{alias}"
    logger.info("Loading model from %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def get_model_run_id(cfg: PipelineConfig) -> str | None:
    """Resolve the MLflow run ID that produced the champion model version.

    Uses the model alias from ``serving.model_alias`` to look up the
    registered version and returns the run that created it, so evaluation
    results can be attached to the model's own run page.

    Args:
        cfg: The validated ``PipelineConfig`` (uses ``mlflow`` and
            ``serving`` sections).

    Returns:
        The training run ID, or None if it cannot be resolved (e.g. the
        registry is unavailable or the alias does not exist yet).
    """
    try:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(
            cfg.mlflow.model_name, cfg.serving.model_alias
        )
        return version.run_id
    except Exception as e:
        logger.warning("Could not resolve model run ID: %s", e)
        return None


def log_evaluation_results(
    cfg: PipelineConfig, metrics: dict[str, float], artifact_paths: list[str]
) -> None:
    """Log evaluation metrics and artifacts to MLflow.

    Logs to the training run that produced the champion model version, so
    the results are reachable from the model's run page. Falls back to a
    new ``evaluation`` run when the training run cannot be resolved —
    logging outside any active run would silently create an orphan
    anonymous run.

    Args:
        cfg: The validated ``PipelineConfig`` (uses the ``mlflow`` section).
        metrics: Dict of metric name to value.
        artifact_paths: Paths of files to log as artifacts (may be empty).
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    run_id = get_model_run_id(cfg)

    if run_id is not None:
        logger.info("Logging evaluation results to training run %s", run_id)
        with mlflow.start_run(run_id=run_id):
            _log_metrics_and_artifacts(metrics, artifact_paths)
    else:
        logger.warning(
            "Training run not resolved; logging evaluation results to a "
            "new 'evaluation' run"
        )
        with mlflow.start_run(run_name="evaluation"):
            _log_metrics_and_artifacts(metrics, artifact_paths)


def _log_metrics_and_artifacts(
    metrics: dict[str, float], artifact_paths: list[str]
) -> None:
    """Log metrics and artifact files to the active MLflow run."""
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    for path in artifact_paths:
        mlflow.log_artifact(path)
        logger.debug("Artifact logged: %s", path)


def load_test_features(
    path: str, data: DataConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Load test split from features.parquet.

    Input contract: the Parquet must contain the featurisation output
    schema — a tz-aware ``timestamp`` column, the ``data.target_col``
    target column, and one column per feature.

    Args:
        path: Path to the features Parquet file.
        data: ``DataConfig`` with ``target_col`` and split dates
            (``train_end``, ``val_end``).

    Returns:
        A tuple ``(X_test, y_test)`` of numpy arrays: ``X_test`` of
        shape (n_test, n_features) and 1-D ``y_test`` of shape
        (n_test,).
    """
    df = pd.read_parquet(path)

    target_col = data.target_col
    _, _, test_mask = get_split_masks(df, data)
    df_test = df.iloc[test_mask]

    y_test = df_test[target_col].values
    feature_cols = [c for c in df_test.columns if c not in [target_col, "timestamp"]]
    X_test = df_test[feature_cols].values

    logger.info(
        "Test set: %s rows, %d features", f"{len(df_test):,}", len(feature_cols)
    )
    return X_test, y_test


def log_results_table(metrics: dict[str, float]) -> None:
    """Log test-set evaluation metrics line by line.

    Args:
        metrics: Dict of metric name to value.
    """
    for name, value in metrics.items():
        logger.info("Test %s: %.4f", name, value)
