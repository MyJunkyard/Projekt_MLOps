"""common/superset.py — paths, manifest, and staleness for the feature superset.

WS0 (stage 4, scheme B): ``make featurise`` materialises ONE feature superset
covering every enabled feature block, retaining warm-up NaNs. Training and
evaluation read it and apply a per-run ``features.select`` subset; neither
stage may import the other, so the artifact contract lives here.

The superset directory holds three sibling files per resolution::

    {superset_dir}/features_hourly.parquet        <- the artifact
    {superset_dir}/features_hourly_schema.json    <- FeatureSchema sidecar
    {superset_dir}/manifest.json                  <- hashes + provenance

All three are REQUIRED for training/evaluation. A missing file is an
actionable error ("run ``make featurise``"), never a silent fallback — a
half-materialised superset must not train a model on the wrong columns.

``materialisation_hash`` fingerprints the configuration that defines WHAT the
superset contains (target, resolution, feature toggles and windows). It
deliberately excludes ``features.select``: editing an experiment selection
must not invalidate the cache. The full ``params.yaml`` hash is still recorded
as provenance (WS6 logs it to MLflow alongside the run).

Strict leaf like ``common/schema.py``: stdlib + pydantic only, so any package
may import it without a layering question.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from src.common.hashing import (
    compute_params_hash,
    compute_sha256_file,
    get_git_commit_hash,
)
from src.config.models import FeatureSelection, PipelineConfig

MODULE_LOGGER_NAME = "src.common.superset"
logger = logging.getLogger(MODULE_LOGGER_NAME)

#: Keys of ``features`` that change the superset content. ``select`` is
#: excluded on purpose (see module docstring); ``weather.derivatives`` is
#: excluded while it remains unimplemented (WS3), so wiring it up later does
#: not retroactively invalidate supersets written before that workstream lands.
_MATERIALISATION_KEYS = (
    "calendar",
    "weather",
    "generation_mix",
    "cross_border_flows",
    "availability_lags",
    "lags",
    "rolling",
    "derivatives",
)


@dataclass(frozen=True)
class SupersetPaths:
    """The three sibling files of one materialised superset.

    Attributes:
        parquet: The feature frame (identifier + target + all features).
        schema: ``FeatureSchema`` sidecar describing every column.
        manifest: Hashes and provenance for the materialisation.
    """

    parquet: Path
    schema: Path
    manifest: Path

    def ensure_dir(self) -> None:
        """Create the superset directory (and parents) if missing."""
        self.parquet.parent.mkdir(parents=True, exist_ok=True)


class SupersetManifest(BaseModel):
    """``manifest.json`` — provenance for one materialised superset.

    Attributes:
        resolution: Data resolution the superset was built at (e.g. ``h``).
        created_at: ISO-8601 UTC timestamp of the materialisation.
        n_rows: Row count of the written parquet.
        n_columns: Column count of the written parquet.
        materialisation_hash: Fingerprint of the content-defining config.
        params_hash: Hash of the whole ``params.yaml`` (provenance only).
        schema_sha256: Digest of the schema sidecar file.
        superset_sha256: Digest of the parquet file.
        raw_csv_sha256: Digest of the raw price CSV, when it exists.
        git_commit: Commit hash at materialisation time, or ``unknown``.
    """

    model_config = ConfigDict(extra="forbid")

    resolution: str
    created_at: str
    n_rows: int
    n_columns: int
    materialisation_hash: str
    params_hash: str
    schema_sha256: str
    superset_sha256: str
    raw_csv_sha256: str | None = None
    git_commit: str

    @field_validator("resolution", "materialisation_hash", "params_hash")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        """Identity fields must never be blank."""
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v


def superset_paths(cfg: PipelineConfig) -> SupersetPaths:
    """Derive the superset file paths for this config.

    Args:
        cfg: Loaded pipeline configuration.

    Returns:
        The parquet/schema/manifest paths under ``data.superset_dir``, named
        after ``temporal.resolution`` so a resolution switch cannot silently
        reuse the previous superset.
    """
    directory = Path(cfg.data.superset_dir)
    stem = f"features_{cfg.temporal.resolution}"
    return SupersetPaths(
        parquet=directory / f"{stem}.parquet",
        schema=directory / f"{stem}_schema.json",
        manifest=directory / "manifest.json",
    )


def _content_view(cfg: PipelineConfig) -> dict[str, Any]:
    """The configuration subset that defines superset content.

    Only ``features`` keys that add columns are included, plus the target and
    resolution. ``features.select`` is excluded by design.
    """
    features = cfg.features.model_dump()
    return {
        "target_col": cfg.data.target_col,
        "resolution": cfg.temporal.resolution,
        "features": {
            key: features.get(key) for key in _MATERIALISATION_KEYS if key in features
        },
    }


def compute_materialisation_hash(cfg: PipelineConfig) -> str:
    """SHA256 over the content-defining configuration (staleness key).

    Stable across ``features.select`` edits so experiment iterations reuse the
    cache; changes whenever a feature block, window, or the target/resolution
    changes.

    Args:
        cfg: Loaded pipeline configuration.

    Returns:
        The hex digest of the canonical JSON encoding of the content view.
    """
    payload = json.dumps(_content_view(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(
    paths: SupersetPaths,
    cfg: PipelineConfig,
    config_path: str = "params.yaml",
    raw_csv_path: Path | None = None,
) -> SupersetManifest:
    """Assemble a manifest for an already-written superset.

    Args:
        paths: Paths of the materialised files (parquet and schema must
            already exist so their digests can be taken).
        cfg: Loaded pipeline configuration.
        config_path: ``params.yaml`` location for the provenance hash.
        raw_csv_path: Raw price CSV to fingerprint when available.

    Returns:
        The populated manifest.

    Raises:
        FileNotFoundError: If the parquet or schema file is missing.
    """
    import pandas as pd

    if not paths.parquet.exists():
        raise FileNotFoundError(f"superset parquet not written: {paths.parquet}")
    if not paths.schema.exists():
        raise FileNotFoundError(f"superset schema sidecar not written: {paths.schema}")

    frame = pd.read_parquet(paths.parquet)
    raw_csv_digest: str | None = None
    if raw_csv_path is not None and Path(raw_csv_path).exists():
        raw_csv_digest = compute_sha256_file(raw_csv_path)

    return SupersetManifest(
        resolution=cfg.temporal.resolution,
        created_at=datetime.now(UTC).isoformat(),
        n_rows=int(len(frame)),
        n_columns=int(frame.shape[1]),
        materialisation_hash=compute_materialisation_hash(cfg),
        params_hash=compute_params_hash(config_path),
        schema_sha256=compute_sha256_file(paths.schema),
        superset_sha256=compute_sha256_file(paths.parquet),
        raw_csv_sha256=raw_csv_digest,
        git_commit=get_git_commit_hash(),
    )


def write_manifest(paths: SupersetPaths, manifest: SupersetManifest) -> None:
    """Write ``manifest.json`` next to the superset.

    Args:
        paths: Superset file paths (the manifest sibling).
        manifest: The manifest to persist.
    """
    paths.ensure_dir()
    paths.manifest.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "Superset manifest written to %s (materialisation_hash=%s)",
        paths.manifest,
        manifest.materialisation_hash[:12],
    )


def read_manifest(paths: SupersetPaths) -> SupersetManifest | None:
    """Read the manifest, tolerating absence or corruption.

    Returns ``None`` (with a WARNING when the file exists but is broken) so
    callers can decide whether that is fatal. Training treats a missing
    manifest as fatal; a stale one only warns.

    Args:
        paths: Superset file paths.

    Returns:
        The parsed manifest, or ``None``.
    """
    if not paths.manifest.exists():
        return None
    try:
        return SupersetManifest.model_validate_json(
            paths.manifest.read_text(encoding="utf-8")
        )
    except (ValueError, OSError) as exc:  # pydantic ValidationError is a ValueError
        logger.warning(
            "Superset manifest %s is corrupt and will be ignored: %s",
            paths.manifest,
            exc,
        )
        return None


def selection_summary(select: FeatureSelection | None, feature_names: list[str]) -> str:
    """Human-readable summary of a resolved selection (log/test helper).

    Args:
        select: The configured selection, or ``None`` for everything.
        feature_names: Resolved feature column names.

    Returns:
        A one-line description such as
        ``groups=[calendar,lags] -> 12 feature columns``.
    """
    if select is None:
        return f"all superset features ({len(feature_names)} columns)"
    groups = ",".join(select.groups) if select.groups else "-"
    columns = ",".join(select.columns) if select.columns else "-"
    return f"groups=[{groups}] columns=[{columns}] -> {len(feature_names)} columns"
