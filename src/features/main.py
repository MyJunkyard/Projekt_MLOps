"""
features/main.py — Featurisation orchestration.

Orchestrates: load raw → calendar/holiday features → lag/rolling
features → derivative features → split → save. The only place in the
features package that knows the run order. Moved verbatim from
``featurise.py`` (Workstream 0 module restructure).
"""

import logging
from pathlib import Path

import pandas as pd

from src.common.logsetup import setup_logging
from src.common.splits import get_split_masks
from src.config import load_config
from src.features.calendar import (
    _get_holiday_dates,
    add_calendar_features,
    add_holiday_proximity_features,
)
from src.features.derivatives import add_derivative_features
from src.features.lags import add_lag_features, add_rolling_features

# Stable module name (not `__name__` — under `python -m` it becomes
# `"__main__"` and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.features.main"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV and parse the timestamp column.

    Input contract: the CSV must contain a ``timestamp`` column
    (produced by the ingest stage, alongside ``price_eur_mwh`` and
    optionally ``load_mw`` / ``is_imputed``).

    Args:
        path: Path to the raw CSV file.

    Returns:
        A DataFrame with ``timestamp`` parsed as datetime (tz-naive —
        pandas parses ISO UTC timestamps to naive datetimes here) and
        the remaining columns as written by ingest.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def train_val_test_split(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test sets based on date boundaries in config.

    Args:
        df: DataFrame with a tz-aware UTC ``timestamp`` column (the
            featurisation output contract: ``timestamp``, target column,
            and all engineered feature columns).
        cfg: Configuration dict with ``data.train_end`` and ``data.val_end``.

    Returns:
        A tuple ``(train_df, val_df, test_df)`` of disjoint DataFrames
        with the same columns as the input.
    """
    train_mask, val_mask, test_mask = get_split_masks(df, cfg)

    train = df[train_mask].copy()
    val = df[val_mask].copy()
    test = df[test_mask].copy()

    logger.info(
        "Train split: %s rows (%s to %s)",
        f"{len(train):,}",
        train["timestamp"].min(),
        train["timestamp"].max(),
    )
    logger.info(
        "Val split: %s rows (%s to %s)",
        f"{len(val):,}",
        val["timestamp"].min(),
        val["timestamp"].max(),
    )
    logger.info(
        "Test split: %s rows (%s to %s)",
        f"{len(test):,}",
        test["timestamp"].min(),
        test["timestamp"].max(),
    )

    return train, val, test


def save_processed_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    processed_path: str,
    reference_path: str,
) -> None:
    """Save concatenated features and reference dataset.

    Args:
        train: Training split DataFrame (must share the same columns as
            ``val`` and ``test``).
        val: Validation split DataFrame.
        test: Test split DataFrame.
        processed_path: Path to save the concatenated features Parquet.
        reference_path: Path to save the train split as reference data.
    """
    processed_path_obj = Path(processed_path)
    processed_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Concatenate all splits
    full = pd.concat([train, val, test], axis=0)
    full.to_parquet(processed_path_obj, index=False)
    logger.info(
        "Saved features: %s (%s rows)",
        processed_path_obj,
        f"{len(full):,}",
    )

    # Save train split as reference for drift detection
    reference_path_obj = Path(reference_path)
    reference_path_obj.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(reference_path_obj, index=False)
    logger.info(
        "Saved reference dataset: %s (%s rows)",
        reference_path_obj,
        f"{len(train):,}",
    )


def main():
    """Orchestrate feature engineering pipeline."""
    cfg = load_config()
    # Configure the *package* logger by explicit name (never `__name__` —
    # under `python -m` that resolves to "__main__" and would bypass the
    # configured handlers; see MODULE_LOGGER_NAME above). Package scope so
    # every sibling module logger (calendar, lags, derivatives, ...)
    # inherits the handlers and level; leaf-scope would leave them
    # unconfigured (effective WARNING, INFO logs silently dropped).
    setup_logging(cfg, logger_name="src.features")
    raw_path = Path(cfg["data"]["raw_path"]) / "entsoe_prices.csv"
    processed_path = cfg["data"]["processed_path"]
    reference_path = cfg["data"]["reference_path"]

    logger.info("Stage: featurisation")
    logger.info("Loading raw data")
    df = load_raw_data(str(raw_path))
    logger.info("Loaded %s rows from %s", f"{len(df):,}", raw_path)
    logger.debug("Raw data columns: %s", list(df.columns))

    # Sort by timestamp to ensure correct lag computation
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Calendar features
    if cfg["features"]["calendar"]["enabled"]:
        logger.info("Adding calendar features")
        # Build the holiday calendar once for the data's year range (padded
        # by one year on each side) and share it across all holiday features
        holidays = _get_holiday_dates(df)
        df = add_calendar_features(df, holidays)
        logger.debug(
            "Added calendar features: hour, day_of_week, month, week_of_year, "
            "is_holiday, is_workday"
        )

        # Holiday proximity features
        logger.info("Adding holiday proximity features")
        df = add_holiday_proximity_features(df, holidays)
        logger.debug(
            "Added holiday proximity features: days_to_next_holiday, "
            "days_since_last_holiday"
        )

    # Lag features
    if cfg["features"]["lags"]["enabled"]:
        periods = cfg["features"]["lags"]["periods"]
        logger.info("Adding lag features for periods: %s", periods)
        df = add_lag_features(df, periods)

        # Rolling features
        logger.info("Adding rolling features")
        df = add_rolling_features(df, cfg["data"]["target_col"])
        logger.debug(
            "Added rolling features: rolling_mean_24h, rolling_std_24h, "
            "rolling_mean_168h"
        )

    # Derivative features
    if cfg["features"]["derivatives"]["enabled"]:
        order = cfg["features"]["derivatives"]["order"]
        smooth_window = cfg["features"]["derivatives"]["smooth_window"]
        logger.info(
            "Adding derivative features (order=%s, smooth_window=%d)",
            order,
            smooth_window,
        )
        df = add_derivative_features(
            df, cfg["data"]["target_col"], order=order, smooth_window=smooth_window
        )

    # Drop rows with NaN (from lag/derivative creation at start of series)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    if before - after > 0:
        logger.info(
            "Dropped %d rows with NaN from lag/derivative creation",
            before - after,
        )

    logger.debug("Feature columns after engineering: %s", list(df.columns))
    logger.info("Splitting into train/val/test")
    train, val, test = train_val_test_split(df, cfg)

    logger.info("Saving processed data")
    save_processed_data(train, val, test, processed_path, reference_path)
    logger.info("Featurisation complete")


if __name__ == "__main__":
    main()
