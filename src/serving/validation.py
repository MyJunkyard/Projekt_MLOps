"""Pure validation helpers for the serving boundary.

The model-specific feature contract comes from the champion run's
``features_schema.json`` MLflow artifact.  Validation remains usable when
that optional metadata is unavailable: name validation is then skipped,
while request shape, row limits, finite values, and configured ranges are
still enforced.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.common.schema import FeatureSchema

MODULE_LOGGER_NAME = "src.serving.validation"
logger = logging.getLogger(MODULE_LOGGER_NAME)


class PredictionInputError(ValueError):
    """Raised when prediction rows violate the serving contract."""


def load_feature_schema(
    run_id: str, artifact_root: str | Path | None = None
) -> FeatureSchema | None:
    """Load the feature schema attached to a training run.

    Args:
        run_id: MLflow run that produced the registered model version.
        artifact_root: Optional local directory containing
            ``config/features_schema.json`` (primarily for tests and
            offline deployments). When absent, MLflow artifact storage is
            used.

    Returns:
        Parsed schema, or ``None`` after a warning when metadata is
        missing or malformed. Metadata failure is intentionally non-fatal.
    """
    try:
        if artifact_root is not None:
            candidates = (
                Path(artifact_root) / "config" / "features_schema.json",
                Path(artifact_root) / "features_schema.json",
            )
            path = next(
                (candidate for candidate in candidates if candidate.exists()), None
            )
            if path is None:
                logger.warning("Feature schema not found under %s", artifact_root)
                return None
            return FeatureSchema.from_file(path)

        client = mlflow.MlflowClient()
        path = client.download_artifacts(
            run_id=run_id,
            artifact_path="config/features_schema.json",
        )
        return FeatureSchema.from_file(path)
    except Exception as exc:  # metadata must never prevent API startup
        logger.warning(
            "Could not load feature schema for run %s: %s; "
            "serving will skip feature-name validation",
            run_id,
            exc,
        )
        return None


def _is_finite_number(value: Any) -> bool:
    """Return whether a value is a finite real number (bools allowed)."""
    if isinstance(value, bool):
        return True
    return isinstance(value, Real) and math.isfinite(float(value))


def _validate_timestamp(value: Any, row_index: int) -> None:
    """Validate an optional timestamp and require timezone awareness."""
    if not isinstance(value, str) and not isinstance(value, (datetime, pd.Timestamp)):
        raise PredictionInputError(
            f"row {row_index}: timestamp must be an ISO-8601 datetime"
        )
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise PredictionInputError(
            f"row {row_index}: timestamp is not parseable: {value!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PredictionInputError(
            f"row {row_index}: timestamp must be timezone-aware"
        )


def validate_feature_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_names: Sequence[str] | None,
    max_rows: int,
    ranges: Mapping[str, Sequence[Real]] | None = None,
) -> list[str] | None:
    """Validate and normalize prediction rows.

    Args:
        rows: JSON feature objects, one per prediction time.
        expected_names: Canonical feature names from the champion schema,
            or ``None`` when schema metadata is unavailable.
        max_rows: Maximum accepted batch size.
        ranges: Optional inclusive numeric bounds keyed by feature name.

    Returns:
        Canonical feature-name order when schema validation is enabled,
        otherwise ``None``.

    Raises:
        PredictionInputError: If any request-level or row-level rule fails.
    """
    if not rows:
        raise PredictionInputError("features must contain at least one row")
    if len(rows) > max_rows:
        raise PredictionInputError(
            f"features contains {len(rows)} rows; maximum is {max_rows}"
        )
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise PredictionInputError("features must be a list of objects")

    expected = list(expected_names) if expected_names is not None else None
    expected_set = set(expected or ())
    bounds = dict(ranges or {})

    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PredictionInputError(
                f"row {row_index}: each feature must be an object"
            )
        keys = set(row)
        if expected is not None:
            feature_keys = keys - {"timestamp"}
            missing = sorted(expected_set - feature_keys)
            extra = sorted(feature_keys - expected_set)
            if missing or extra:
                details = []
                if missing:
                    details.append(f"missing {missing}")
                if extra:
                    details.append(f"unexpected {extra}")
                raise PredictionInputError(
                    f"row {row_index}: feature columns do not match the model "
                    f"schema ({'; '.join(details)})"
                )
        for name, value in row.items():
            if name == "timestamp":
                _validate_timestamp(value, row_index)
                continue
            if not _is_finite_number(value):
                raise PredictionInputError(
                    f"row {row_index}: feature {name!r} must be a finite number"
                )
            if name in bounds:
                lower, upper = (float(bound) for bound in bounds[name])
                if not lower <= float(value) <= upper:
                    raise PredictionInputError(
                        f"row {row_index}: feature {name!r}={value!r} is outside "
                        f"the allowed range [{lower}, {upper}]"
                    )
    return expected
