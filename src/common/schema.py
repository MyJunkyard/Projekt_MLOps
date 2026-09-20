"""
common/schema.py — Column metadata registry (``ColumnSpec``/``FeatureSchema``).

The structured counterpart to flat column names. Feature-engineering
blocks *declare* what they produce as ``ColumnSpec`` records; the
resulting ``FeatureSchema`` travels alongside the DataFrame (saved as
``features_schema.json`` next to ``features.parquet``) so downstream
consumers (training loader, serving validation, plots, contract tests)
can ask questions like "which columns are availability-lagged generation
features?" **without parsing name patterns**.

Design rules:

- **Flat names stay opaque identifiers.** Nothing in this module
  interprets name structure; semantics live in the metadata fields
  (``role``, ``group``, ``availability_lag_hours``, ``derived_from``).
- **Strict leaf.** Imports only stdlib + pydantic — same layering as
  ``config/models.py``; every package may depend on it.
- **Drift guard.** :meth:`FeatureSchema.assert_matches_dataframe` pins
  the schema to the actual frame columns, so a block that adds a column
  without declaring it (or declares one it fails to add) fails fast.

Pydantic v2 preserves ``dict`` insertion order, so ``columns`` doubles
as the canonical column order.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

MODULE_LOGGER_NAME = "src.common.schema"
logger = logging.getLogger(MODULE_LOGGER_NAME)


class ColumnRole(str, Enum):
    """Semantic role of a column in a pipeline DataFrame."""

    IDENTIFIER = "identifier"  # row key (timestamp) — never a feature
    TARGET = "target"  # prediction target — never a feature
    FEATURE = "feature"  # model input
    META = "meta"  # bookkeeping (e.g. is_imputed) — excluded from X


class ColumnSpec(BaseModel):
    """Metadata for one column of a pipeline DataFrame.

    Attributes:
        name: Column name (the flat string used in the DataFrame).
        role: Semantic role (see :class:`ColumnRole`).
        group: Feature family that produced the column (``"base"``,
            ``"calendar"``, ``"lag"``, ``"rolling"``, ``"weather"``,
            ``"generation"``, ``"availability_lag"``, ``"derivative"``,
            ``"external"``, ...).
        dtype: Pandas/numpy dtype string of the column values.
        availability_lag_hours: Real-time publication lag of the
            underlying measurement in hours (0 = current value known at
            prediction time). Non-zero marks deployment-consistent
            lagged externals (e.g. ``load_mw_lag1h``).
        derived_from: Names of the raw/source columns this column was
            computed from (empty for base columns).
        description: Human-readable explanation (optional).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    role: ColumnRole
    group: str = "base"
    dtype: str = "float64"
    availability_lag_hours: int = Field(default=0, ge=0)
    derived_from: list[str] = Field(default_factory=list)
    description: str = ""


class FeatureSchema(BaseModel):
    """Ordered registry of :class:`ColumnSpec` for one DataFrame.

    The single source of truth for "what is in ``features.parquet`` and
    what does each column mean". Saved next to the parquet as
    ``features_schema.json`` and logged to MLflow with every training
    run so serving-time validation can key on metadata instead of name
    patterns.
    """

    model_config = ConfigDict(extra="forbid")

    columns: dict[str, ColumnSpec]

    # -- lookups ----------------------------------------------------------

    def names(self) -> list[str]:
        """All column names in canonical order."""
        return list(self.columns)

    def feature_names(self) -> list[str]:
        """Names of columns with role ``FEATURE`` (the model matrix X)."""
        return [
            name
            for name, spec in self.columns.items()
            if spec.role == ColumnRole.FEATURE
        ]

    def select(
        self,
        role: ColumnRole | None = None,
        group: str | None = None,
    ) -> list[str]:
        """Names matching the given role and/or group (AND semantics).

        Args:
            role: Restrict to this role (``None`` = any).
            group: Restrict to this group (``None`` = any).

        Returns:
            Matching column names in canonical order.
        """
        return [
            name
            for name, spec in self.columns.items()
            if (role is None or spec.role == role)
            and (group is None or spec.group == group)
        ]

    # -- drift guard -------------------------------------------------------

    def assert_matches_dataframe(self, df: pd.DataFrame) -> None:
        """Raise unless the schema and the frame agree on columns.

        The runtime drift guard: every schema column must exist in the
        frame (a declared-but-absent column means a block lied about its
        output) and every frame column must be declared (an undeclared
        column would silently escape the metadata registry).

        Args:
            df: The DataFrame the schema describes.

        Raises:
            ValueError: Listing exactly which columns are missing from
                the frame vs. undeclared in the schema.
        """
        frame_cols = set(df.columns)
        schema_cols = set(self.columns)
        missing = sorted(schema_cols - frame_cols)
        undeclared = sorted(frame_cols - schema_cols)
        if missing or undeclared:
            raise ValueError(
                "FeatureSchema does not match the DataFrame — "
                f"declared but absent from frame: {missing}; "
                f"present in frame but undeclared: {undeclared}. "
                "Fix the feature block that produced the mismatch."
            )

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the schema as JSON; returns the written path."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Feature schema written to %s", path_obj)
        return path_obj

    @classmethod
    def from_file(cls, path: str | Path) -> "FeatureSchema":
        """Load a schema from a JSON file written by :meth:`save`.

        Raises:
            FileNotFoundError: If the file does not exist.
            pydantic.ValidationError: If the JSON does not match the model.
        """
        path_obj = Path(path)
        return cls.model_validate_json(path_obj.read_text(encoding="utf-8"))

    def to_json(self) -> str:
        """Serialize to a JSON string (``model_dump_json`` alias)."""
        return self.model_dump_json(indent=2)


def schema_from_dataframe(
    df: pd.DataFrame,
    *,
    target_col: str,
    timestamp_col: str = "timestamp",
    meta_cols: tuple[str, ...] = ("is_imputed",),
) -> FeatureSchema:
    """Build a minimal schema for a frame with no engineered features.

    Convenience for tests and for bootstrapping: declares the timestamp
    identifier, the target, known meta columns, and treats every other
    column as an ``external`` feature. Production feature groups always
    declare their columns explicitly via blocks in
    ``features.main.build_features``.

    Args:
        df: The DataFrame to describe.
        target_col: Name of the target column.
        timestamp_col: Name of the identifier column.
        meta_cols: Column names to declare as ``META`` when present.

    Returns:
        A :class:`FeatureSchema` covering exactly ``df.columns``.
    """
    columns: dict[str, ColumnSpec] = {}
    columns[timestamp_col] = ColumnSpec(
        name=timestamp_col,
        role=ColumnRole.IDENTIFIER,
        group="base",
        dtype="datetime64[ns, UTC]",
    )
    columns[target_col] = ColumnSpec(
        name=target_col, role=ColumnRole.TARGET, group="base", dtype="float64"
    )
    for col in meta_cols:
        if col in df.columns:
            columns[col] = ColumnSpec(
                name=col, role=ColumnRole.META, group="imputation", dtype="bool"
            )
    for col in df.columns:
        if col not in columns:
            columns[col] = ColumnSpec(
                name=col, role=ColumnRole.FEATURE, group="external"
            )
    return FeatureSchema(columns=columns)


def load_schema_or_none(path: str | Path) -> FeatureSchema | None:
    """Load a schema file, returning ``None`` (with a WARNING) on failure.

    Graceful-read helper for consumers that can operate without the
    schema (backward compatibility with pre-WS4 parquets): a missing or
    malformed file must never crash the caller.

    Args:
        path: Candidate ``features_schema.json`` path.

    Returns:
        The loaded :class:`FeatureSchema`, or ``None`` if the file is
        missing or invalid.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    try:
        return FeatureSchema.from_file(path_obj)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "Feature schema file %s is invalid (%s) — ignoring it",
            path_obj,
            exc,
        )
        return None
