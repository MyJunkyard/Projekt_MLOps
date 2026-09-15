"""
common/splits.py — Train/val/test split-mask logic.

Moved verbatim from ``utils.py`` (Workstream 0 module restructure): the
chronological split policy has a single source of truth, shared by
featurisation, training, and evaluation.
"""

import numpy as np
import pandas as pd


def get_split_masks(
    df: pd.DataFrame, cfg: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for train/val/test splits based on config dates.

    Required DataFrame contract:
        - ``timestamp``: timezone-aware datetime column (UTC), one row
          per period, no duplicate timestamps.

    Args:
        df: DataFrame meeting the contract above.
        cfg: Configuration dict with ``data.train_end`` and ``data.val_end``
            boundaries (tz-naive date strings, interpreted as UTC).

    Returns:
        A tuple ``(train_mask, val_mask, test_mask)`` of boolean numpy
        arrays, each of shape (n_samples,), mutually exclusive and
        collectively covering all rows of ``df``.
    """
    # Boundaries are tz-naive in params.yaml; data is tz-aware (UTC)
    train_end = pd.Timestamp(cfg["data"]["train_end"], tz="UTC")
    val_end = pd.Timestamp(cfg["data"]["val_end"], tz="UTC")

    train_mask = (df["timestamp"] < train_end).to_numpy()
    val_mask = ((df["timestamp"] >= train_end) & (df["timestamp"] < val_end)).to_numpy()
    test_mask = (df["timestamp"] >= val_end).to_numpy()
    return train_mask, val_mask, test_mask
