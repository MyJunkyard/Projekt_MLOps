"""
training/loader.py — Dynamic model loading and feature-array loading.

Moved verbatim from ``train.py`` (Workstream 0 module restructure).
"""

import importlib
import logging
import subprocess

import pandas as pd

from src.common.splits import get_split_masks
from src.config.models import DataConfig, ModelConfig

MODULE_LOGGER_NAME = "src.training.loader"
logger = logging.getLogger(MODULE_LOGGER_NAME)


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

    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c not in [target_col, "timestamp"]]
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
        A list of feature column names (excluding target and timestamp).
    """
    df = pd.read_parquet(path)
    target_col = data.target_col
    feature_cols = [c for c in df.columns if c not in [target_col, "timestamp"]]
    return feature_cols


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
