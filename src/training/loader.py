"""
training/loader.py — Dynamic model loading and feature-array loading.

Feature-column selection (Workstream 4): when a
``features_schema.json`` sidecar (written by the featurise stage next
to ``features.parquet``) exists, feature columns come from its
``role == FEATURE`` entries — the explicit, versioned metadata
registry. Without the sidecar (pre-WS4 parquets), the loader falls
back to the legacy "all non-timestamp, non-target columns" rule with a
WARNING, so old artifacts keep working but the gap is visible.
"""

import importlib
import logging
import subprocess
from pathlib import Path

import pandas as pd

from src.common.schema import FeatureSchema, load_schema_or_none
from src.common.splits import get_split_masks
from src.config.models import DataConfig, ModelConfig

MODULE_LOGGER_NAME = "src.training.loader"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def _feature_columns(features_path: str, target_col: str) -> list[str]:
    """Resolve feature columns from the schema sidecar (legacy fallback).

    Args:
        features_path: Path to the features Parquet file.
        target_col: Name of the target column.

    Returns:
        Feature column names in canonical order.
    """
    schema_path = Path(features_path).with_name("features_schema.json")
    schema: FeatureSchema | None = load_schema_or_none(schema_path)
    if schema is not None:
        return schema.feature_names()
    logger.warning(
        "No feature schema sidecar at %s — falling back to 'all "
        "non-timestamp, non-target columns' (pre-Workstream-4 parquet). "
        "Re-run `python -m src featurise` to get the explicit schema.",
        schema_path,
    )
    df = pd.read_parquet(features_path)
    return [c for c in df.columns if c not in [target_col, "timestamp"]]


def load_model(model_cfg: ModelConfig):
    """Dynamically load a model class from its fully qualified name.

    Example: "sklearn.dummy.DummyRegressor" → sklearn.dummy.DummyRegressor

    Args:
        model_cfg: ``ModelConfig`` with ``type`` (dotted path) and
            ``params`` (constructor kwargs).

    Returns:
        An instantiated model object.

    Raises:
        ImportError: If the module path cannot be imported.
        AttributeError: If the class name is not found in the module.
    """
    module_path, class_name = model_cfg.type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(**model_cfg.params)


def load_features(path: str, data: DataConfig) -> tuple:
    """Load features.parquet and separate features (X) from target (y).

    Args:
        path: Path to the features Parquet file.
        data: ``DataConfig`` with ``target_col`` and split dates.

    Returns:
        A tuple ``(X_train, y_train, X_val, y_val, X_test, y_test)`` of
        numpy arrays.
    """
    df = pd.read_parquet(path)

    # Separate target
    target_col = data.target_col
    y = df[target_col].values

    # Feature columns from the schema sidecar (legacy fallback inside)
    feature_cols = _feature_columns(path, target_col)
    X = df[feature_cols].values

    # Reconstruct splits from the concatenated data using shared split logic
    train_mask, val_mask, test_mask = get_split_masks(df, data)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info("Features: %d columns", len(feature_cols))
    logger.debug("X_train: %s, y_train: %s", X_train.shape, y_train.shape)
    logger.debug("X_val: %s, y_val: %s", X_val.shape, y_val.shape)
    logger.debug("X_test: %s, y_test: %s", X_test.shape, y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_feature_names(path: str, data: DataConfig) -> list[str]:
    """Get the list of feature column names from the features parquet.

    Args:
        path: Path to the features Parquet file.
        data: ``DataConfig`` with ``target_col``.

    Returns:
        A list of feature column names (from the schema sidecar when
        present; legacy all-non-timestamp/non-target fallback otherwise).
    """
    return _feature_columns(path, data.target_col)


def get_git_commit_hash() -> str:
    """Get the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
