"""
training/loader.py — model loading and the split-feature assembly (WS4 D4-h).

WS0 (stage 4, scheme B): training reads the materialised feature superset
(``{superset_dir}/features_{resolution}.parquet``) instead of a pre-split
``features.parquet``. All three superset files (parquet, schema sidecar,
manifest) are REQUIRED — a missing artifact is an actionable error pointing at
``make featurise``, never a silent fallback, because a half-materialised
superset must not train a model on the wrong columns.

The superset covers every enabled feature block. Each experiment narrows it
via ``features.select`` (groups and/or explicit columns), so the dropna
policy is *per selection*: rows are dropped only where the selected feature
columns (or the target) are NaN. That is what lets a selection without
weekly-scale lags keep rows a union-dropna artifact would discard.
"""

import importlib
import logging

import pandas as pd

from src.common.hashing import get_git_commit_hash
from src.common.schema import FeatureSchema
from src.common.selection import (
    drop_selected_nan_rows,
    open_superset,
    resolve_selection,
)
from src.common.splits import get_split_masks
from src.common.superset import selection_summary
from src.config.models import ModelConfig, PipelineConfig

MODULE_LOGGER_NAME = "src.training.loader"
logger = logging.getLogger(MODULE_LOGGER_NAME)

__all__ = [
    "get_git_commit_hash",
    "load_model",
    "load_superset_features",
]


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


def load_superset_features(
    cfg: PipelineConfig,
) -> tuple[
    list[pd.DataFrame],
    list[pd.DataFrame],
    list[str],
    FeatureSchema,
    pd.DataFrame,
]:
    """Open the superset, apply the selection, and build the split matrices.

    Args:
        cfg: Loaded pipeline configuration.

    Returns:
        ``(X_splits, y_splits, feature_cols, subset_schema, clean_frame)``
        where ``X_splits``/``y_splits`` are ``[train, val, test]`` and
        ``subset_schema`` is the superset schema narrowed to the selection
        (logged to MLflow as the per-run model-instance link, decision D-c).
    """
    frame, schema = open_superset(cfg)
    feature_cols = resolve_selection(schema, cfg.features.select)
    logger.info(
        "Feature selection: %s",
        selection_summary(cfg.features.select, feature_cols),
    )

    clean = drop_selected_nan_rows(frame, feature_cols, cfg.data.target_col)
    subset_schema = schema.subset(feature_cols)

    train_mask, val_mask, test_mask = get_split_masks(clean, cfg.data)
    X = clean[feature_cols]
    y = clean[cfg.data.target_col]
    X_splits = [X[train_mask], X[val_mask], X[test_mask]]
    y_splits = [y[train_mask], y[val_mask], y[test_mask]]

    for name, part in zip(("train", "val", "test"), X_splits, strict=True):
        logger.debug("X_%s: %s", name, part.shape)

    return X_splits, y_splits, feature_cols, subset_schema, clean
