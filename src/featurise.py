"""
featurise.py — Stage 2: Feature engineering for energy price forecasting.

Loads raw CSV, adds calendar (including Polish public holidays), lag, rolling,
and derivative features, splits into train/val/test, and saves as Parquet.
Gated by params.yaml feature flags.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_split_masks, load_config, setup_logging

# Stable module name (not `__name__` — under `python -m` it becomes
# `"__main__"` and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.featurise"
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


def add_calendar_features(
    df: pd.DataFrame, holidays: pd.DatetimeIndex
) -> pd.DataFrame:
    """Add calendar-based features from timestamp column.

    Features added:
    - hour: 0-23
    - day_of_week: 0=Monday, 6=Sunday
    - month: 1-12
    - week_of_year: 1-53
    - is_holiday: True for Polish public holidays
    - is_workday: not weekend AND not holiday

    Args:
        df: DataFrame with a ``timestamp`` datetime column (the
            featurisation input contract: ``timestamp``, target column
            from ingest).
        holidays: Sorted holiday midnights (UTC) from ``_get_holiday_dates``.

    Returns:
        The same DataFrame (mutated in place and returned) with the
        calendar feature columns ``hour``, ``day_of_week``, ``month``,
        ``week_of_year``, ``is_holiday``, ``is_workday`` added.
    """
    timestamp = df["timestamp"]
    df["hour"] = timestamp.dt.hour
    df["day_of_week"] = timestamp.dt.dayofweek
    df["month"] = timestamp.dt.month
    df["week_of_year"] = timestamp.dt.isocalendar().week.astype(int)

    # Polish public holidays
    df = add_holiday_features(df, holidays)

    # Workday = weekday (Mon-Fri) and not a holiday
    df["is_workday"] = (df["day_of_week"] < 5) & (~df["is_holiday"])

    return df


def _get_holiday_dates(df: pd.DataFrame, country: str = "Poland") -> pd.DatetimeIndex:
    """Build a sorted DatetimeIndex of public holiday midnights (UTC).

    The calendar covers the data's year range padded by one year on each
    side, so proximity features are correct for timestamps at the very
    start/end of the data (e.g. Dec 31 needs the next year's New Year).

    This is the single place where the holiday calendar is built; all
    holiday feature functions receive it precomputed (review point 6).

    Args:
        df: DataFrame with a ``timestamp`` datetime column (its year
            range, padded ±1 year, defines the calendar span).
        country: Country name for the holiday calendar (default: "Poland").

    Returns:
        Sorted, deduplicated ``DatetimeIndex`` of holiday midnights (UTC).

    Raises:
        ValueError: If the country has no supported holiday calendar.
    """
    from workalendar.europe import Poland

    calendars = {"Poland": Poland}
    if country not in calendars:
        raise ValueError(f"Unsupported holiday calendar country: {country!r}")
    cal = calendars[country]()

    years = df["timestamp"].dt.year
    holiday_dates: set = set()
    for year in range(years.min() - 1, years.max() + 2):
        for date, _ in cal.holidays(year):
            holiday_dates.add(date)

    return pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in sorted(holiday_dates)])


def add_holiday_features(df: pd.DataFrame, holidays: pd.DatetimeIndex) -> pd.DataFrame:
    """Add a public holiday flag using a precomputed holiday calendar.

    Holidays include: New Year, Easter Monday, Labour Day, Constitution Day,
    Corpus Christi, Assumption, All Saints, Independence Day, Christmas Day,
    and Second Day of Christmas.

    Args:
        df: DataFrame with a ``timestamp`` datetime column.
        holidays: Sorted holiday midnights (UTC) from ``_get_holiday_dates``.

    Returns:
        The same DataFrame (mutated in place and returned) with the
        boolean ``is_holiday`` column added.
    """
    df["is_holiday"] = df["timestamp"].dt.normalize().isin(holidays)

    return df


def add_holiday_proximity_features(
    df: pd.DataFrame, holidays: pd.DatetimeIndex
) -> pd.DataFrame:
    """Add holiday proximity features using a precomputed holiday calendar.

    Features added:
    - days_to_next_holiday: integer days until the next upcoming holiday
    - days_since_last_holiday: integer days since the most recent past holiday

    Both are computed on whole calendar days, so every hour of a day gets
    the same value (e.g. 23:00 on the day before a holiday gives 1, not 0 —
    review point 5). On a holiday itself both are 0. If there is no holiday
    in range at all, the value is NaN rather than an ambiguous 0.

    These capture price spikes that often appear in the days before a long
    weekend.

    Args:
        df: DataFrame with a ``timestamp`` datetime column.
        holidays: Sorted holiday midnights (UTC) from ``_get_holiday_dates``.

    Returns:
        The same DataFrame (mutated in place and returned) with nullable
        integer columns ``days_to_next_holiday`` and
        ``days_since_last_holiday`` added (NaN when no holiday is in
        range).
    """
    # Whole calendar days as proleptic Gregorian ordinals — immune to the
    # Timedelta.days truncation that affects sub-day remainders.
    ts_days = np.array(
        [d.toordinal() for d in df["timestamp"].dt.date], dtype=np.int64
    )
    holiday_days = np.array([d.toordinal() for d in holidays.date], dtype=np.int64)

    # Vectorised binary search: O(n log m) instead of a full scan per
    # timestamp (review point 5). side="right" - 1 gives the last holiday
    # on or before ts; side="left" gives the first holiday on or after ts.
    # On a holiday both point at the holiday itself, yielding 0 by
    # construction.
    last_idx = np.searchsorted(holiday_days, ts_days, side="right") - 1
    next_idx = np.searchsorted(holiday_days, ts_days, side="left")

    days_since = np.full(len(df), np.nan)
    has_last = last_idx >= 0
    days_since[has_last] = ts_days[has_last] - holiday_days[last_idx[has_last]]

    days_to_next = np.full(len(df), np.nan)
    has_next = next_idx < len(holiday_days)
    days_to_next[has_next] = holiday_days[next_idx[has_next]] - ts_days[has_next]

    df["days_to_next_holiday"] = pd.Series(days_to_next, index=df.index).astype(
        "Int64"
    )
    df["days_since_last_holiday"] = pd.Series(days_since, index=df.index).astype(
        "Int64"
    )

    return df


def add_lag_features(
    df: pd.DataFrame, periods: list[int]
) -> pd.DataFrame:
    """Add lag features of the target variable.

    Lags are computed on the full dataset before splitting to avoid
    data leakage. Rows with NaN from lag creation at the beginning
    of the series are dropped.

    Args:
        df: DataFrame with the target column ``price_eur_mwh``, sorted
            by ``timestamp`` (lags are order-dependent).
        periods: Lag periods in hours, e.g. [1, 2, 3, 24, 48, 168].

    Returns:
        The same DataFrame (mutated in place and returned) with float
        lag columns ``lag_{p}h`` added (NaN for the first ``p`` rows).
    """
    for period in periods:
        df[f"lag_{period}h"] = df["price_eur_mwh"].shift(period)
    return df


def add_rolling_features(
    df: pd.DataFrame, target_col: str, windows: list[int] | None = None
) -> pd.DataFrame:
    """Add rolling statistics of the target variable.

    Rolling features are computed on the full dataset before splitting to
    avoid data leakage. Uses trailing windows (no future data).

    Args:
        df: DataFrame containing ``target_col``, sorted by ``timestamp``
            (rolling windows are order-dependent).
        target_col: Name of the target column to compute rolling stats on.
        windows: List of window sizes in hours.

    Returns:
        The same DataFrame (mutated in place and returned) with float
        columns ``rolling_mean_{w}h`` and ``rolling_std_{w}h`` added for
        each window ``w``.
    """
    if windows is None:
        windows = [24, 168]
    for window in windows:
        df[f"rolling_mean_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).std()
        )

    return df


def add_derivative_features(
    df: pd.DataFrame,
    target_col: str,
    order: list[int] | None = None,
    smooth_window: int = 3,
) -> pd.DataFrame:
    """Add derivative (difference) features of the target variable.

    Raw price derivatives amplify noise. Before computing:
    1. Apply a short rolling mean (window=smooth_window) to smooth
    2. Then compute diff(1) for first derivative, diff(2) for second

    Conceptually: first derivative = price velocity (rising/falling)
    Second derivative = price acceleration (speeding up/slowing down)

    Args:
        df: DataFrame containing ``target_col``, sorted by ``timestamp``
            (differences are order-dependent).
        target_col: Name of the target column.
        order: List of derivative orders to compute (1, 2, or both).
        smooth_window: Rolling mean window before differencing.

    Returns:
        The same DataFrame (mutated in place and returned) with float
        columns ``{target_col}_diff_{n}`` added for each order ``n``.
    """
    # Smooth the target to reduce noise
    if order is None:
        order = [1, 2]
    smoothed = df[target_col].rolling(window=smooth_window, min_periods=1).mean()

    for n in order:
        df[f"{target_col}_diff_{n}"] = smoothed.diff(n)

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
    setup_logging(cfg, logger_name=MODULE_LOGGER_NAME)
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
