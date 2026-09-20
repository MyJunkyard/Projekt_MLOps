"""
features/main.py — Featurisation orchestration.

Orchestrates: load raw → calendar/holiday features → lag/rolling
features → derivative features → split → save. The only place in the
features package that knows the run order. Moved verbatim from
``featurise.py`` (Workstream 0 module restructure).
"""

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.common.logsetup import setup_logging
from src.common.splits import get_split_masks
from src.config import load_config
from src.config.models import DataConfig, FeaturesConfig
from src.features.calendar import (
    _get_holiday_dates,
    add_calendar_features,
    add_holiday_proximity_features,
)
from src.features.derivatives import add_derivative_features
from src.features.lags import add_lag_features, add_rolling_features
from src.ingestion.weather import load_weather_cache, merge_weather

# Stable module name (not `__name__` — under `python -m` it becomes
# `"__main__"` and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.features.main"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def _apply_step(
    df: pd.DataFrame, label: str, fn: Callable[[pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """Run one feature-group step with uniform logging.

    Standard block mechanics for every group in ``build_features``: INFO-log
    the step label, apply ``fn``, DEBUG-log exactly the columns it added.
    The *dispatch* stays explicit per group (``if enabled`` blocks in run
    order) — order is load-bearing for the no-leakage guarantee, so it is
    kept visible rather than hidden in a registry. New groups (WS3/WS4)
    add one standardized block in their documented position.

    Args:
        df: Input frame.
        label: Human-readable step description, INFO-logged before running.
        fn: Pure transformation applied to ``df``.

    Returns:
        The transformed DataFrame.
    """
    logger.info("Adding %s", label)
    before = set(df.columns)
    df = fn(df)
    logger.debug("Added %s: %s", label, sorted(set(df.columns) - before))
    return df


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
    df: pd.DataFrame, data: DataConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test sets based on date boundaries in config.

    Args:
        df: DataFrame with a tz-aware UTC ``timestamp`` column (the
            featurisation output contract: ``timestamp``, target column,
            and all engineered feature columns).
        data: ``DataConfig`` with the ``train_end`` and ``val_end``
            boundaries.

    Returns:
        A tuple ``(train_df, val_df, test_df)`` of disjoint DataFrames
        with the same columns as the input.
    """
    train_mask, val_mask, test_mask = get_split_masks(df, data)

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


def build_features(
    df: pd.DataFrame,
    features: FeaturesConfig,
    target_col: str,
    weather: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Apply all enabled feature groups to the full frame (pre-split).

    Pure transformation (no I/O): sorts by ``timestamp``, applies each
    enabled feature group in run order, and drops rows left as NaN by
    lag/derivative creation at the series start. The caller splits the
    returned frame into train/val/test.

    Lag/rolling features are computed here on the **full frame before
    splitting** — the no-leakage ordering (see ``features.lags`` and
    ``TestNoLeakageAcrossSplits``). The weather merge (WS3) lands in
    this function, before the lag block; the generation-mix merge
    (WS4) will take the same position.

    Block convention: every group is an explicit ``if enabled`` block in
    run order (order is load-bearing, kept visible) whose mechanics go
    through ``_apply_step`` (uniform INFO/DEBUG logging with an exact
    added-columns diff). New groups add one block in their documented
    position — no registry, no dispatch indirection.

    Args:
        df: Raw DataFrame with ``timestamp`` and ``target_col`` columns.
        features: ``FeaturesConfig`` with the per-group toggles/settings.
        target_col: Name of the target column to derive lags, rolling
            statistics, and derivatives from.
        weather: Optional per-location weather frames (``timestamp`` +
            float variable columns), keyed by location name. Required
            when ``features.weather.enabled`` is true; the caller loads
            them from the cache (never the network — decision D2).

    Returns:
        The featurised DataFrame with no NaN rows.

    Raises:
        ValueError: If ``features.weather.enabled`` is true but no
            weather data was supplied, if a configured location is
            missing from ``weather``, or if every row is dropped as NaN
            (e.g. the longest configured lag period exceeds the
            available history).
    """
    # Sort by timestamp to ensure correct lag computation
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Calendar features
    if features.calendar.enabled:
        # Build the holiday calendar once for the data's year range (padded
        # by one year on each side) and share it across all holiday features
        holidays = _get_holiday_dates(df)
        df = _apply_step(
            df,
            "calendar features",
            lambda d: add_calendar_features(d, holidays),
        )
        df = _apply_step(
            df,
            "holiday proximity features",
            lambda d: add_holiday_proximity_features(d, holidays),
        )

    # Weather features (Workstream 3): merge before the lag block so the
    # no-leakage ordering is preserved. One standardized block per
    # location; the merge itself normalizes both join keys to naive UTC.
    if features.weather.enabled:
        if not weather:
            raise ValueError(
                "features.weather.enabled is true but no weather data was "
                "supplied — load the weather cache in features.main() and "
                "pass it as `weather`."
            )
        for location in features.weather.locations:
            if location not in weather:
                raise ValueError(
                    f"features.weather.locations includes {location!r} but "
                    "no weather frame was supplied for it — check the "
                    "cache load in features.main()."
                )
            df = _apply_step(
                df,
                f"weather features for {location}",
                lambda d, loc=location, frame=weather[location]: merge_weather(
                    d, frame, loc
                ),
            )

    # Lag + rolling features
    if features.lags.enabled:
        periods = features.lags.periods
        windows = features.lags.rolling_windows
        df = _apply_step(
            df,
            f"lag features for periods: {periods}",
            lambda d: add_lag_features(d, target_col, periods),
        )
        df = _apply_step(
            df,
            f"rolling features for windows: {windows}",
            lambda d: add_rolling_features(d, target_col, windows),
        )

    # Derivative features
    if features.derivatives.enabled:
        order = features.derivatives.order
        smooth_window = features.derivatives.smooth_window
        df = _apply_step(
            df,
            f"derivative features (order={order}, smooth_window={smooth_window})",
            lambda d: add_derivative_features(
                d, target_col, order=order, smooth_window=smooth_window
            ),
        )

    # Drop rows with NaN (from lag/derivative creation at start of series)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(
            "Dropped %d rows with NaN from lag/derivative creation",
            dropped,
        )
    if df.empty:
        longest_lag = max(features.lags.periods) if features.lags.enabled else None
        hint = (
            f" The longest configured lag period is {longest_lag}h, which "
            "exceeds the available history — shorten "
            "features.lags.periods or provide more raw data."
            if longest_lag is not None
            else ""
        )
        raise ValueError(
            "Feature engineering produced an empty DataFrame "
            f"(dropped all {before} rows as NaN from lag/derivative "
            f"creation).{hint}"
        )

    return df


def main():
    """Orchestrate feature engineering pipeline."""
    cfg = load_config()
    # Configure the *package* logger by explicit name (never `__name__` —
    # under `python -m` that resolves to "__main__" and would bypass the
    # configured handlers; see MODULE_LOGGER_NAME above). Package scope so
    # every sibling module logger (calendar, lags, derivatives, ...)
    # inherits the handlers and level; leaf-scope would leave them
    # unconfigured (effective WARNING, INFO logs silently dropped).
    setup_logging(cfg.logging, logger_name="src.features")
    raw_path = Path(cfg.data.raw_path) / "entsoe_prices.csv"
    processed_path = cfg.data.processed_path
    reference_path = cfg.data.reference_path

    logger.info("Stage: featurisation")
    logger.info("Loading raw data")
    df = load_raw_data(str(raw_path))
    logger.info("Loaded %s rows from %s", f"{len(df):,}", raw_path)
    logger.debug("Raw data columns: %s", list(df.columns))

    # Weather (Workstream 3): the ingest stage owns all network I/O and
    # persists the per-location cache (decision D2); featurise only reads
    # it — never the network.
    weather: dict[str, pd.DataFrame] | None = None
    if cfg.features.weather.enabled:
        logger.info(
            "Loading weather cache for locations: %s",
            cfg.features.weather.locations,
        )
        weather = {
            location: load_weather_cache(
                cfg.data.raw_path,
                location,
                df["timestamp"].min(),
                df["timestamp"].max(),
                cfg.features.weather.variables,
                allow_synthetic=cfg.features.weather.allow_synthetic,
            )
            for location in cfg.features.weather.locations
        }
    else:
        logger.debug("Weather features disabled — no weather merge")

    logger.info("Engineering features")
    df = build_features(df, cfg.features, cfg.data.target_col, weather=weather)

    logger.debug("Feature columns after engineering: %s", list(df.columns))
    logger.info("Splitting into train/val/test")
    train, val, test = train_val_test_split(df, cfg.data)

    logger.info("Saving processed data")
    save_processed_data(train, val, test, processed_path, reference_path)
    logger.info("Featurisation complete")


if __name__ == "__main__":
    main()
