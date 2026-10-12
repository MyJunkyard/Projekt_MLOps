"""common/schema_artifact.py — the per-run feature-schema MLflow artifact.

WS4 (stage 4, step 0 / decision D4-h): ``read_feature_schema`` and the
``SCHEMA_ARTIFACT_NAME`` constant lived in ``training/registry.py``, which forced
``evaluation.reporting`` to import ``training.registry``. They are the *read half*
of the model-instance link (WS0 decision D-c): the writer is ``registry`` (it
logs ``config/features_schema.json`` with every run), and the reader is used by
evaluation to resolve a model's input columns and — per its original docstring —
by serving. Putting the reader in ``common`` lets both ``evaluation`` and
``serving`` read a run's schema **without importing ``training.registry``**, so
the ``training ⇄ evaluation`` import cycle WS4 would otherwise introduce never
forms. ``registry`` imports the constant back from here (its write path).
"""

import logging
from pathlib import Path

import mlflow

from src.common.schema import FeatureSchema

MODULE_LOGGER_NAME = "src.common.schema_artifact"
logger = logging.getLogger(MODULE_LOGGER_NAME)

__all__ = ["SCHEMA_ARTIFACT_NAME", "read_feature_schema"]

#: Artifact name (inside MLflow's ``config/`` artifact directory) for the
#: per-run feature schema. WS0: the resolved *subset* schema, i.e. this run's
#: exact model input contract (decision D-c). Written by ``training.registry``;
#: read back here for evaluation and serving.
SCHEMA_ARTIFACT_NAME = "features_schema.json"


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
