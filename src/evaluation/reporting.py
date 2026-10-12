"""
evaluation/reporting.py — MLflow result reporting and test-feature loading.

Moved verbatim from ``evaluate.py`` (Workstream 0 module restructure).
"""

import logging

import mlflow
import numpy as np
import pandas as pd

from src.common.schema import FeatureSchema
from src.common.superset import selection_summary
from src.config.models import PipelineConfig
from src.training.loader import resolve_selection
from src.training.registry import read_feature_schema

MODULE_LOGGER_NAME = "src.evaluation.reporting"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def load_model_from_registry(cfg: PipelineConfig, run_id: str | None = None):
    """Load a model from MLflow — the champion alias by default, or a given run.

    With no ``run_id`` (default) the champion alias from ``serving.model_alias``
    is used. An alias points to exactly one registered version, so the loaded
    model is unambiguous. The alias stays the serving-time "live pointer" to
    the current champion.

    When ``run_id`` is given (WS6 D10), the model is loaded from that run's own
    artifact via ``runs:/{run_id}/model``. This is what lets a non-champion run
    — a baseline, or one arm of EXP-022 — be scored on the test split.
    (``runs:/`` addresses an unregistered run's model artifact; a
    ``models:/<run_id>`` URI would misread the id as a registered model name
    and fail.)

    Args:
        cfg: The validated ``PipelineConfig`` (uses ``mlflow`` and
            ``serving`` sections).
        run_id: Optional MLflow run ID to score directly, bypassing the alias.

    Returns:
        A loaded MLflow pyfunc model.
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    if run_id is not None:
        model_uri = f"runs:/{run_id}/model"
    else:
        alias = cfg.serving.model_alias
        model_uri = f"models:/{cfg.mlflow.model_name}@{alias}"
    logger.info("Loading model from %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def get_model_run_id(cfg: PipelineConfig) -> str | None:
    """Resolve the MLflow run ID that produced the champion model version.

    Uses the model alias from ``serving.model_alias`` to look up the
    registered version and returns the run that created it, so evaluation
    results can be attached to the model's own run page. This is the *single*
    alias→run_id resolver; callers that already hold a run id (an explicit
    ``run_id`` override) pass it through directly instead of calling here.

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
    cfg: PipelineConfig,
    metrics: dict[str, float],
    artifact_paths: list[str],
    run_id: str | None = None,
) -> None:
    """Log evaluation metrics and artifacts to MLflow.

    Logs to the training run that produced the champion model version, so
    the results are reachable from the model's run page. Falls back to a
    new ``evaluation`` run when the training run cannot be resolved —
    logging outside any active run would silently create an orphan
    anonymous run.

    When ``run_id`` is given (WS6 D10) the metrics are attached to *that*
    run instead — so scoring a non-champion run via ``make evaluate RUN_ID=``
    lands on the run that was actually scored, never the champion. ``None``
    keeps the default champion-alias resolution.

    Args:
        cfg: The validated ``PipelineConfig`` (uses the ``mlflow`` section).
        metrics: Dict of metric name to value.
        artifact_paths: Paths of files to log as artifacts (may be empty).
        run_id: Optional MLflow run ID to attach the results to; ``None``
            resolves the champion run via the alias.
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    if run_id is None:
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


def resolve_evaluate_columns(
    cfg: PipelineConfig, schema: FeatureSchema, run_id: str | None = None
) -> tuple[list[str], FeatureSchema]:
    """Resolve which feature columns the scored model expects.

    WS0 decision D-c: the model-instance link is the *subset schema logged
    with the run*, not a copy of the data. Evaluation therefore reads the
    scored run's ``config/features_schema.json`` artifact and uses exactly
    those columns, falling back to the config ``features.select`` when the
    artifact cannot be found.

    With no ``run_id`` (default) the champion run is resolved via the alias;
    when ``run_id`` is given (WS6 D10) the columns come from *that* run's own
    schema artifact, so a ``RUN_ID=`` evaluation reads the scored run's input
    contract — never the champion's, never the config's.

    Args:
        cfg: The validated ``PipelineConfig`` (uses ``mlflow``/``serving``
            sections and ``features.select``).
        schema: The full superset schema read from disk.
        run_id: Optional MLflow run ID whose schema to read; ``None``
            resolves the champion run via the alias.

    Returns:
        A tuple ``(feature_cols, subset_schema)`` where ``feature_cols`` is
        ordered canonically and ``subset_schema`` is the schema restricted to
        those columns.

    Raises:
        ValueError: If the run's columns are absent from the current superset
            (the superset was rebuilt without a needed feature block, or the
            scored model was trained against a different superset).
    """
    available = set(schema.feature_names())
    run_schema: FeatureSchema | None = None
    if run_id is None:
        run_id = get_model_run_id(cfg)

    if run_id is not None:
        run_schema = read_feature_schema(run_id, artifact_root=None)

    if run_schema is not None:
        feature_cols = run_schema.feature_names()
        logger.info(
            "Feature columns taken from run %s (%d columns)",
            run_id,
            len(feature_cols),
        )
    else:
        feature_cols = resolve_selection(schema, cfg.features.select)
        logger.warning(
            "No feature schema on run %s; falling back to config "
            "features.select (%s)",
            run_id,
            selection_summary(cfg.features.select, feature_cols),
        )

    missing = [c for c in feature_cols if c not in available]
    if missing:
        raise ValueError(
            f"The champion model expects {len(missing)} column(s) that are not "
            f"in the current superset: {missing}. Either the superset was "
            "rebuilt without the relevant feature block (re-run `make "
            "featurise` with it enabled) or the champion model was trained "
            "against a different superset (re-train)."
        )

    subset = schema.subset(feature_cols)
    logger.info(
        "Evaluation selection: %s", selection_summary(cfg.features.select, feature_cols)
    )
    return subset.feature_names(), subset


def load_test_features(
    frame: pd.DataFrame, feature_cols: list[str], target_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Select the model's feature columns and the target from a clean frame.

    Args:
        frame: Row-aligned frame (superset restricted to the resolved
            selection, already NaN-filtered — see ``resolve_evaluate_columns``).
        feature_cols: Resolved feature columns (the model's input contract).
        target_col: Name of the target column.

    Returns:
        A tuple ``(X_test, y_test)`` of numpy arrays: ``X_test`` of shape
        (n_test, n_features) and 1-D ``y_test`` of shape (n_test,).

    Raises:
        ValueError: If any resolved feature column is absent from the frame.
    """
    missing = [c for c in feature_cols if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Test frame is missing {len(missing)} feature column(s) the model "
            f"was trained on: {missing}. The superset on disk no longer "
            "contains them — re-run `make featurise` with the relevant feature "
            "block enabled, or re-train against the current superset."
        )

    y_test = frame[target_col].values
    X_test = frame[feature_cols].values

    logger.info(
        "Test set: %s rows, %d features", f"{len(frame):,}", len(feature_cols)
    )
    return X_test, y_test


def log_results_table(metrics: dict[str, float]) -> None:
    """Log test-set evaluation metrics line by line.

    Args:
        metrics: Dict of metric name to value.
    """
    for name, value in metrics.items():
        logger.info("Test %s: %.4f", name, value)
