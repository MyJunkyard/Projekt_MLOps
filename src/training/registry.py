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

from src.common.hashing import compute_params_hash, get_git_commit_hash
from src.common.schema import FeatureSchema
from src.config.models import PipelineConfig

MODULE_LOGGER_NAME = "src.training.registry"
logger = logging.getLogger(MODULE_LOGGER_NAME)

#: Artifact name (inside MLflow's ``config/`` artifact directory) for the
#: per-run feature schema. WS0: the resolved *subset* schema, i.e. this run's
#: exact model input contract (decision D-c). Read back by
#: ``read_feature_schema`` for evaluation and by serving.
SCHEMA_ARTIFACT_NAME = "features_schema.json"


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
    cfg: PipelineConfig,
    feature_names: list[str] | None = None,
    run_name: str | None = None,
    promote_to_production: bool = False,
    feature_schema: FeatureSchema | None = None,
) -> str:
    """Log model, params, and metrics to MLflow.

    Logs:
    - Model parameters (from ``model.params``)
    - Model type and feature flags
    - Temporal config (resolution, horizon)
    - Metrics (rmse, mae, mape, r2, etc.)
    - Feature importances as a JSON artifact (if available)
    - The resolved subset feature schema as ``config/features_schema.json``
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
        cfg: The validated ``PipelineConfig``.
        feature_names: List of feature column names (for feature importances).
        feature_schema: The RESOLVED SUBSET ``FeatureSchema`` actually trained
            on. Logged as ``config/features_schema.json`` so serving and
            evaluation can resolve this model's exact input contract from its
            own run (WS0 decision D-c). When ``None`` the artifact is skipped
            with a warning.
        run_name: Optional name for the MLflow run.
        promote_to_production: Whether to set the champion alias on the
            newly registered version. Defaults to False (safe default —
            baselines are registered but never promoted).
        feature_schema: WS0 subset schema — the model's exact input
            contract for this run, logged so serving/evaluation can resolve
            columns from the run's own artifacts (decision D-c).

    Returns:
        The MLflow run ID as a string.
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    model_name = cfg.mlflow.model_name

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log model parameters
        for key, value in cfg.model.params.items():
            mlflow.log_param(key, value)

        # Log model type
        mlflow.log_param("model_type", cfg.model.type)

        # Log feature flags (serialized view of the feature groups — the
        # flag names become MLflow param names, so they must stay flat)
        for group, settings in cfg.features.model_dump().items():
            if isinstance(settings, dict) and "enabled" in settings:
                mlflow.log_param(f"feature_{group}", settings["enabled"])

        # Log temporal config
        mlflow.log_param("resolution", cfg.temporal.resolution)
        mlflow.log_param("horizon", cfg.temporal.horizon)

        # Log params hash for quick run comparison
        params_hash = compute_params_hash()
        mlflow.set_tag("params_hash", params_hash)

        # Log full params.yaml as artifact for reproducibility
        params_path = Path("params.yaml")
        if params_path.exists():
            mlflow.log_artifact(str(params_path), artifact_path="config")

        # Log the raw-data manifest so runs link to an exact raw-data
        # snapshot (Workstream 5), plus its own hash as a tag.
        raw_manifest_path = Path(cfg.data.raw_path) / "manifest.json"
        if raw_manifest_path.exists():
            mlflow.log_artifact(str(raw_manifest_path), artifact_path="config")
            try:
                manifest_sha = json.loads(
                    raw_manifest_path.read_text(encoding="utf-8")
                ).get("sha256")
                if manifest_sha:
                    mlflow.set_tag("manifest_sha256", manifest_sha)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read %s: %s", raw_manifest_path, exc)

        # Log the feature schema sidecar: the metadata registry describing
        # every column of the superset. WS0 logs the RESOLVED SUBSET schema
        # (the per-run model-instance link, decision D-c) rather than the
        # whole-superset file, so serving/evaluation can resolve a model's
        # exact input contract from its own run's artifacts.
        if feature_schema is not None:
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                f.write(feature_schema.model_dump_json(indent=2))
                temp_path = f.name

            mlflow.log_artifact(temp_path, artifact_path="config")
            Path(temp_path).unlink()
            logger.info(
                "Logged subset feature schema (%d columns) as config/%s",
                len(feature_schema.columns),
                SCHEMA_ARTIFACT_NAME,
            )
        else:
            logger.warning(
                "No feature schema supplied — run consumers cannot resolve "
                "this run's input columns from artifacts."
            )

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
        promote = promote_to_production and cfg.mlflow.promote_to_production
        if promote and model_info.registered_model_version:
            client = mlflow.MlflowClient()
            alias = cfg.mlflow.champion_alias
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
                cfg.mlflow.promote_to_production,
            )

        # Log feature importances if available
        if feature_names is not None:
            log_feature_importances(model, feature_names)

        # Log tags
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.set_tag("stage", "3")

        logger.debug("MLflow tracking URI: %s", cfg.mlflow.tracking_uri)
        logger.debug("MLflow experiment: %s", cfg.mlflow.experiment_name)
        logger.info("MLflow run ID: %s", run_id)

    return run_id


def read_feature_schema(
    run_id: str | None, artifact_root: str | Path | None = None
) -> FeatureSchema | None:
    """Read a run's feature schema artifact (the model-instance link).

    The single artifact *reader*: evaluation resolves the champion run's input
    columns with it, and serving delegates here so "what does a model expect"
    has exactly one implementation.

    Resolution order:

    1. If ``artifact_root`` is given, look for ``config/features_schema.json``
       beneath it (a directory, or a downloaded artifact dir).
    2. Otherwise ask MLflow for run ``run_id``.

    Args:
        run_id: MLflow run ID, or ``None`` to only use ``artifact_root``.
        artifact_root: Optional local artifact directory (short-circuits MLflow).

    Returns:
        The parsed schema, or ``None`` when it cannot be found or read —
        metadata lookups must never crash a caller (serving starts up anyway).
    """
    try:
        if artifact_root is not None:
            root = Path(artifact_root)
            candidates = [
                root / "config" / SCHEMA_ARTIFACT_NAME,
                root / SCHEMA_ARTIFACT_NAME,
            ]
            local_path = next((c for c in candidates if c.exists()), None)
            if local_path is None:
                logger.warning(
                    "Feature schema not found under %s (looked in %s)",
                    root,
                    ", ".join(str(c) for c in candidates),
                )
                return None
            return FeatureSchema.from_file(local_path)

        if not run_id:
            return None

        remote_path = mlflow.MlflowClient().download_artifacts(
            run_id, f"config/{SCHEMA_ARTIFACT_NAME}"
        )
        return FeatureSchema.from_file(remote_path)
    except Exception as exc:  # metadata must never break a caller
        logger.warning(
            "Could not load feature schema for run %s: %s",
            run_id,
            exc,
        )
        return None
