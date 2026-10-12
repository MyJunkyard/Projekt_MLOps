"""
training/registry.py — MLflow logging, registration, and promotion (WS4 D4-h).

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
from src.common.schema_artifact import SCHEMA_ARTIFACT_NAME
from src.common.superset import read_manifest, superset_paths
from src.config.models import PipelineConfig

MODULE_LOGGER_NAME = "src.training.registry"
logger = logging.getLogger(MODULE_LOGGER_NAME)

#: Stage-4 tree. Bumped per stage; Stage-2/3 runs keep their own ``stage``
#: value (WS6 decision D9). Written by code — never authorable in config.
STAGE = "4"


def _log_param_once(key: str, value: object, seen: set[str]) -> None:
    """Log one MLflow param, refusing to overwrite an already-logged key.

    ``model.params`` is a free-form dict flattened into the same namespace as
    the pipeline's own params, and MLflow raises on a duplicate key with a
    different value — so a collision must fail here, with a message naming
    both keys, rather than deep inside the tracking store.

    Args:
        key: Param name (already section-prefixed by the caller).
        value: Scalar param value (lists must be pre-joined by the caller).
        seen: Keys already logged in this run.

    Raises:
        ValueError: If ``key`` was already logged in this run.
    """
    if key in seen:
        raise ValueError(
            f"MLflow param {key!r} would be logged twice in one run — "
            "rename the colliding model.params entry (see "
            "docs/contracts.md §6 for the param naming convention)."
        )
    seen.add(key)
    mlflow.log_param(key, value)


def _join(values: list) -> str:
    """Comma-join a list of scalars for a flat MLflow param (WS6 D2).

    ``str(list)`` would render ``"[3, 6, 12]"`` — unreadable in the compare
    table and un-sortable — so ``order: [1, 2]`` becomes ``"1,2"`` and
    ``smooth_windows: [3, 6, 12]`` becomes ``"3,6,12"``.

    Args:
        values: The list of scalar values (ints, strings, ...).

    Returns:
        The comma-joined string (``""`` for an empty list).
    """
    return ",".join(str(v) for v in values)


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
    row_counts: dict[str, int] | None = None,
) -> str:
    """Log model, params, and metrics to MLflow.

    Logs:
    - Model parameters (from ``model.params``)
    - Model type and temporal config (resolution, horizon)
    - WS6 experiment params: temporal strategy, derivative order/smoothing
      windows (comma-joined), the resolved column selection
      (``select_groups``/``select_columns_n``), the superset contents
      (``superset_groups``) and — only when ``feature_schema`` is passed — the
      run's own selection (``feature_groups``, ``n_features``). Every param
      goes through :func:`_log_param_once` (WS6 D1) so a ``model.params``
      collision fails loudly here, not inside MLflow.
    - Per-split row counts as ``n_rows_train/_val/_test`` (WS6 D6) when given
    - ``run_tags`` from config as MLflow tags (WS6 D4); ``stage`` (``STAGE``)
    - The resolved subset feature schema as ``config/features_schema.json``
    - Full ``params.yaml``, the raw-data manifest (``config/manifest.json`` +
      ``manifest_sha256`` tag) and the superset manifest
      (``config/superset/manifest.json`` + ``superset_path`` /
      ``superset_materialisation_hash`` tags) for provenance (WS6 D5)
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
            with a warning and the schema-derived params (``feature_groups``,
            ``n_features``) are omitted.
        row_counts: Optional ``{"train": n, "val": n, "test": n}`` of rows per
            split *after* the NaN/warm-up policy — the number the model
            actually saw (WS6 D6). Logged as ``n_rows_train/_val/_test`` so a
            silent row-set shift is visible in the compare table. ``None``
            logs no row-count params.
        run_name: Optional name for the MLflow run.
        promote_to_production: Whether to set the champion alias on the
            newly registered version. Defaults to False (safe default —
            baselines are registered but never promoted).

    Returns:
        The MLflow run ID as a string.
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    model_name = cfg.mlflow.model_name

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # WS6 (D4): apply run_tags immediately, before any other tag, so the
        # EXPERIMENTS.md convention keys are the first thing written. An
        # invalid key never reaches here — MlflowConfig.run_tags is validated
        # at load (RESERVED_RUN_TAGS) — so there is no WARNING branch.
        for tag_key, tag_value in cfg.mlflow.run_tags.items():
            mlflow.set_tag(tag_key, str(tag_value))

        # WS6 (D1): every param flows through one guard so a model.params key
        # colliding with a pipeline-prefixed key fails here with a clear
        # message, instead of inside MLflow's "Changing param values" error.
        seen: set[str] = set()

        # Log model parameters (verbatim — model.params is a free-form dict).
        for key, value in cfg.model.params.items():
            _log_param_once(key, value, seen)

        # Log model type and temporal config (existing keys — Stage-2/3 runs
        # depend on them, so they keep their names).
        _log_param_once("model_type", cfg.model.type, seen)
        _log_param_once("resolution", cfg.temporal.resolution, seen)
        _log_param_once("horizon", cfg.temporal.horizon, seen)

        # WS6 experiment params: the settings that define an experiment, as
        # flat scalars so the compare table can filter/sort on them (§2 table).
        # Lists are comma-joined (D2); weather-derivative params are logged
        # unconditionally (the block is inert in config but WS3 will select on
        # it, keeping EXP-011 comparable without a WS6 follow-up).
        _log_param_once("temporal_strategy", cfg.temporal.strategy, seen)
        _log_param_once("derivative_order", _join(cfg.features.derivatives.order), seen)
        _log_param_once(
            "derivative_smooth_windows",
            _join(cfg.features.derivatives.smooth_windows),
            seen,
        )
        _log_param_once(
            "weather_derivative_order",
            _join(cfg.features.weather.derivatives.order),
            seen,
        )
        _log_param_once(
            "weather_derivative_smooth_windows",
            _join(cfg.features.weather.derivatives.smooth_windows),
            seen,
        )

        # Resolved column selection (the config echo of this run's subset).
        select = cfg.features.select
        if select is None:
            _log_param_once("select_groups", "all", seen)
            _log_param_once("select_columns_n", 0, seen)
        else:
            groups = _join(select.groups) if select.groups else "-"
            _log_param_once("select_groups", groups, seen)
            _log_param_once("select_columns_n", len(select.columns), seen)

        # Superset contents (config claim, D8) — replaces the old six
        # feature_<group> booleans, which were constant under materialise-all
        # and fully redundant with the manifest + materialisation_hash (D5).
        # Only groups with ``enabled=True`` count; ``availability_lags`` (a
        # plain dict) and ``select`` (None/FeatureSelection) have no ``enabled``
        # key and are skipped.
        superset_groups = sorted(
            group
            for group, settings in cfg.features.model_dump().items()
            if isinstance(settings, dict) and settings.get("enabled") is True
        )
        _log_param_once("superset_groups", _join(superset_groups), seen)

        # The run's own selection — only when a subset schema was passed
        # (D7): logging n_features/feature_groups on a baseline would claim an
        # input contract the model does not have.
        if feature_schema is not None:
            _log_param_once(
                "n_features", len(feature_schema.feature_names()), seen
            )
            feature_groups = sorted(
                {spec.group for spec in feature_schema.columns.values()}
            )
            _log_param_once("feature_groups", _join(feature_groups), seen)

        # Per-split row counts (D6): the number of rows the model actually
        # saw, after the warm-up/NaN policy. Absent argument → params absent.
        if row_counts is not None:
            for split, count in row_counts.items():
                _log_param_once(f"n_rows_{split}", int(count), seen)

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

        # Log the superset manifest (WS6 D5): the provenance of the feature
        # superset this run trained on. Nested under ``config/superset/`` so
        # its ``manifest.json`` basename does not overwrite the raw-data
        # ``config/manifest.json`` (same basename, verified collision).
        # Missing or corrupt manifest → WARNING only: provenance must never
        # fail a trained run (``read_manifest`` returns ``None`` for both).
        superset_paths_for_run = superset_paths(cfg)
        superset_manifest = read_manifest(superset_paths_for_run)
        if superset_manifest is not None:
            mlflow.log_artifact(
                str(superset_paths_for_run.manifest), artifact_path="config/superset"
            )
            mlflow.set_tag("superset_path", str(superset_paths_for_run.parquet))
            mlflow.set_tag(
                "superset_materialisation_hash",
                superset_manifest.materialisation_hash,
            )
        else:
            logger.warning(
                "No superset manifest found at %s — run provenance tags "
                "(superset_path, superset_materialisation_hash) are skipped.",
                superset_paths_for_run.manifest,
            )

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

        # Log tags. ``stage`` is a module constant (WS6 D9): written by code,
        # never authorable in config; Stage-2/3 runs keep their own value.
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.set_tag("stage", STAGE)

        logger.debug("MLflow tracking URI: %s", cfg.mlflow.tracking_uri)
        logger.debug("MLflow experiment: %s", cfg.mlflow.experiment_name)
        logger.info("MLflow run ID: %s", run_id)

    return run_id

