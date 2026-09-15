"""
training/registry.py — MLflow logging, registration, and promotion.

Moved verbatim from ``train.py`` (Workstream 0 module restructure): the
MLflow mechanics are a separate role from the training orchestration.
"""

import json
import logging
from pathlib import Path

import mlflow
from mlflow.sklearn import log_model

from src.common.hashing import compute_params_hash
from src.training.loader import get_git_commit_hash

MODULE_LOGGER_NAME = "src.training.registry"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def log_feature_importances(model, feature_names: list[str]) -> None:
    """Log model feature importances as a JSON artifact to MLflow.

    Args:
        model: A fitted model with a ``feature_importances_`` attribute.
        feature_names: List of feature column names.
    """
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    if len(importances) == 0:
        return

    importance_dict = {
        name: float(imp)
        for name, imp in zip(feature_names, importances, strict=False)
    }

    # Sort by importance descending
    importance_dict = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    # Log as JSON artifact
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(importance_dict, f, indent=2)
        temp_path = f.name

    mlflow.log_artifact(temp_path, artifact_path="feature_importances")
    Path(temp_path).unlink()


def log_to_mlflow(
    model,
    metrics: dict[str, float],
    cfg: dict,
    feature_names: list[str] | None = None,
    run_name: str | None = None,
    promote_to_production: bool = False,
) -> str:
    """Log model, params, and metrics to MLflow.

    Logs:
    - Model parameters (from ``cfg["model"]["params"]``)
    - Model type and feature flags
    - Temporal config (resolution, horizon)
    - Metrics (rmse, mae, mape, r2, etc.)
    - Feature importances as a JSON artifact (if available)
    - Full ``params.yaml`` as an artifact (for reproducibility)
    - Tags: ``params_hash`` (SHA256 fingerprint), ``git_commit``, ``stage``

    If ``promote_to_production`` is True and the model was registered, the
    new version is set as the champion alias (``mlflow.champion_alias``).
    Baselines must NOT be promoted — only the primary model becomes the
    deployment target. The promotion is additionally gated by
    ``mlflow.promote_to_production`` in the config.

    Args:
        model: A fitted scikit-learn-compatible model.
        metrics: Dict of metric name to value.
        cfg: Configuration dict with MLflow and model settings.
        feature_names: List of feature column names (for feature importances).
        run_name: Optional name for the MLflow run.
        promote_to_production: Whether to set the champion alias on the
            newly registered version. Defaults to False (safe default —
            baselines are registered but never promoted).

    Returns:
        The MLflow run ID as a string.
    """
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    model_name = cfg["mlflow"]["model_name"]

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log model parameters
        for key, value in cfg["model"]["params"].items():
            mlflow.log_param(key, value)

        # Log model type
        mlflow.log_param("model_type", cfg["model"]["type"])

        # Log feature flags
        for group, settings in cfg["features"].items():
            if isinstance(settings, dict) and "enabled" in settings:
                mlflow.log_param(f"feature_{group}", settings["enabled"])

        # Log temporal config
        mlflow.log_param("resolution", cfg["temporal"]["resolution"])
        mlflow.log_param("horizon", cfg["temporal"]["horizon"])

        # Log params hash for quick run comparison
        params_hash = compute_params_hash()
        mlflow.set_tag("params_hash", params_hash)

        # Log full params.yaml as artifact for reproducibility
        params_path = Path("params.yaml")
        if params_path.exists():
            mlflow.log_artifact(str(params_path), artifact_path="config")

        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Log model artifact. Use cloudpickle serialization: MLflow 3.x
        # defaults to skops, which refuses to serialize XGBoost models and
        # the custom baseline classes unless they are explicitly whitelisted.
        model_info = log_model(
            sk_model=model,
            name="model",
            registered_model_name=model_name,
            serialization_format="cloudpickle",
        )

        # Promote to champion alias (only the primary model; baselines are
        # registered but never promoted). An alias points to exactly one
        # version, so multiple promotions cannot accumulate the way registry
        # stages did.
        promote = promote_to_production and cfg["mlflow"].get(
            "promote_to_production", True
        )
        if promote and model_info.registered_model_version:
            client = mlflow.MlflowClient()
            alias = cfg["mlflow"].get("champion_alias", "champion")
            client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=str(model_info.registered_model_version),
            )
            logger.info(
                "Model version %s set as '%s' alias",
                model_info.registered_model_version,
                alias,
            )
        else:
            logger.debug(
                "Model version %s registered without promotion "
                "(promote_to_production=%s, config=%s)",
                model_info.registered_model_version,
                promote_to_production,
                cfg["mlflow"].get("promote_to_production", True),
            )

        # Log feature importances if available
        if feature_names is not None:
            log_feature_importances(model, feature_names)

        # Log tags
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.set_tag("stage", "2")

        logger.debug("MLflow tracking URI: %s", cfg["mlflow"]["tracking_uri"])
        logger.debug("MLflow experiment: %s", cfg["mlflow"]["experiment_name"])
        logger.info("MLflow run ID: %s", run_id)

    return run_id
