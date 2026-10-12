"""common/selection.py — superset access, column selection, and per-subset dropna.

WS4 (stage 4, step 0 / decision D4-h): the superset *read* path and the
``features.select`` resolution lived in ``training/loader.py``, which forced the
``evaluation`` package to import ``training`` (``evaluation.main`` for
``open_superset``/``drop_selected_nan_rows``, ``evaluation.reporting`` for
``resolve_selection``). WS4 adds the reverse edge — ``training.main`` calls the
residual analysis in ``evaluation`` — so a bidirectional ``training ⇄
evaluation`` package coupling was about to form. These functions are superset
*data-access* primitives, not model-fitting, so they move here: ``common`` is the
neutral home (the artifact *contract* already lives in ``common/superset.py``;
this is its read/select counterpart).

``training`` and ``evaluation`` both import from here; neither imports the other.
``common/superset.py`` is a strict stdlib+pydantic leaf, but this module reads
parquet, so it is a *sibling* that may import pandas — the layering rule it
respects is "no intra-package cycle", which it satisfies.
"""

import logging
from pathlib import Path

import pandas as pd

from src.common.schema import FeatureSchema
from src.common.superset import (
    SupersetManifest,
    compute_materialisation_hash,
    read_manifest,
    superset_paths,
)
from src.config.models import FeatureSelection, PipelineConfig

MODULE_LOGGER_NAME = "src.common.selection"
logger = logging.getLogger(MODULE_LOGGER_NAME)

__all__ = [
    "GROUP_ALIASES",
    "drop_selected_nan_rows",
    "open_superset",
    "resolve_selection",
]

#: Group-name aliases so ``features.select.groups: [lags]`` matches the
#: schema's finer-grained ``lag``/``rolling`` groups. Values are matched
#: against ``ColumnSpec.group``.
GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "lags": ("lag", "rolling"),
    "derivatives": ("derivative",),
}


def resolve_selection(
    schema: FeatureSchema, select: FeatureSelection | None
) -> list[str]:
    """Resolve ``features.select`` against the superset schema.

    Args:
        schema: Schema of the materialised superset.
        select: The configured selection, or ``None`` to take every feature
            column in the superset.

    Returns:
        Selected feature column names in canonical (schema) order.

    Raises:
        ValueError: If a requested group or column is absent from the
            superset — e.g. a selection naming ``weather`` when the superset
            was materialised with ``weather.enabled: false``. The message
            lists what IS available so the fix is obvious.
    """
    available = schema.feature_names()
    if select is None:
        return available

    wanted: set[str] = set()
    available_groups = {spec.group for spec in schema.columns.values()}
    for group in select.groups:
        targets = GROUP_ALIASES.get(group, (group,))
        matched = [t for t in targets if t in available_groups]
        if not matched:
            raise ValueError(
                f"features.select.groups: group '{group}' is not in the "
                f"superset. Available groups: {sorted(available_groups)}. "
                "Enable the feature block in params.yaml and re-run "
                "`make featurise` to materialise it."
            )
        for group_name in matched:
            wanted.update(schema.select(group=group_name))

    for column in select.columns:
        if column not in available:
            raise ValueError(
                f"features.select.columns: column '{column}' is not in the "
                f"superset ({len(available)} feature columns available). "
                "Check the name, or re-run `make featurise` if the block is "
                "enabled but not yet materialised."
            )
        wanted.add(column)

    return [name for name in available if name in wanted]


def _require_superset_file(path: Path, description: str) -> None:
    """Raise an actionable error when a superset file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature superset {description} not found at {path}. Run "
            "`make featurise` (python -m src featurise) to materialise it — "
            "training reads the superset and does not rebuild features."
        )


def _warn_if_stale(cfg: PipelineConfig, manifest: SupersetManifest | None) -> None:
    """WARN (never raise) when the superset predates the current config."""
    if manifest is None:
        logger.warning(
            "Superset manifest is missing or unreadable — staleness cannot be "
            "verified. Run `make featurise` to refresh it."
        )
        return
    current = compute_materialisation_hash(cfg)
    if manifest.materialisation_hash != current:
        logger.warning(
            "Superset looks STALE: manifest materialisation_hash=%s but the "
            "current config hashes to %s (feature blocks, windows, target, or "
            "resolution changed since it was written). Run `make featurise` "
            "to regenerate it.",
            manifest.materialisation_hash[:12],
            current[:12],
        )


def open_superset(cfg: PipelineConfig) -> tuple[pd.DataFrame, FeatureSchema]:
    """Open the materialised superset and validate it against its sidecar.

    Args:
        cfg: Loaded pipeline configuration.

    Returns:
        The superset frame and its schema.

    Raises:
        FileNotFoundError: If the parquet, schema sidecar, or manifest is
            missing.
        ValueError: If the sidecar does not match the frame's columns.
    """
    paths = superset_paths(cfg)
    _require_superset_file(paths.parquet, "parquet")
    _require_superset_file(paths.schema, "schema sidecar")
    _require_superset_file(paths.manifest, "manifest")

    frame = pd.read_parquet(paths.parquet)
    schema = FeatureSchema.from_file(paths.schema)
    schema.assert_matches_dataframe(frame)

    _warn_if_stale(cfg, read_manifest(paths))
    logger.info(
        "Opened superset %s (%s rows, %s columns)",
        paths.parquet,
        f"{len(frame):,}",
        frame.shape[1],
    )
    return frame, schema


def drop_selected_nan_rows(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> pd.DataFrame:
    """Drop rows NaN in the SELECTED features or the target (per-subset dropna).

    Rows whose NaN lives only in *unselected* columns are kept — the point of
    the superset design (a calendar-only selection ignores lag warm-up).

    Args:
        df: Superset frame (or a split of it).
        feature_cols: Resolved selection.
        target_col: Name of the target column.

    Returns:
        A filtered copy.

    Raises:
        ValueError: If the selection is empty, or every row is dropped.
    """
    if not feature_cols:
        raise ValueError(
            "Selection resolved to zero feature columns — check "
            "features.select against the superset schema."
        )
    out = df.dropna(subset=feature_cols + [target_col]).copy()
    if out.empty:
        raise ValueError(
            f"After dropping NaN rows for the {len(feature_cols)} selected "
            "feature column(s), no rows remain. Loosen the selection, or "
            "re-run `make featurise` on a longer history."
        )
    if len(out) < len(df):
        logger.info(
            "Dropped %s/%s rows NaN in the selected features",
            f"{len(df) - len(out):,}",
            f"{len(df):,}",
        )
    return out

