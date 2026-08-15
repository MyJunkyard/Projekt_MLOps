"""
featurise.py — Stage 1: Feature engineering for energy price forecasting.

Loads raw CSV, adds calendar and lag features, splits into train/val/test,
and saves as Parquet. Gated by params.yaml feature flags.
"""

from pathlib import Path

import pandas as pd

from src.utils import get_split_masks, load_config


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV and parse the timestamp column.

    Args:
        path: Path to the raw CSV file.

    Returns:
        A DataFrame with ``timestamp`` (datetime) and ``price_eur_mwh``
        (float) columns.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features from timestamp column.

    Features added:
    - hour: 0-23
    - day_of_week: 0=Monday, 6=Sunday
    - month: 1-12
    - week_of_year: 1-53
    - is_holiday: False for synthetic data (placeholder for Stage 2)
    - is_workday: not weekend AND not holiday

    Args:
        df: DataFrame with a ``timestamp`` column.

    Returns:
        A copy of ``df`` with calendar feature columns added.
    """
    timestamp = df["timestamp"]
    df["hour"] = timestamp.dt.hour
    df["day_of_week"] = timestamp.dt.dayofweek
    df["month"] = timestamp.dt.month
    df["week_of_year"] = timestamp.dt.isocalendar().week.astype(int)

    # Synthetic data has no real holidays — placeholder
    df["is_holiday"] = False

    # Workday = weekday (Mon-Fri) and not a holiday
    df["is_workday"] = (df["day_of_week"] < 5) & (~df["is_holiday"])

    return df


def add_lag_features(
    df: pd.DataFrame, periods: list[int]
) -> pd.DataFrame:
    """Add lag features of the target variable.

    Lags are computed on the full dataset before splitting to avoid
    data leakage. Rows with NaN from lag creation at the beginning
    of the series are dropped.

    Args:
        df: DataFrame with a ``price_eur_mwh`` column, sorted by timestamp.
        periods: Lag periods in hours, e.g. [1, 2, 3, 24, 48, 168].

    Returns:
        A copy of ``df`` with lag columns ``lag_{p}h`` added.
    """
    for period in periods:
        df[f"lag_{period}h"] = df["price_eur_mwh"].shift(period)
    return df


def train_val_test_split(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test sets based on date boundaries in config.

    Args:
        df: DataFrame with a tz-aware ``timestamp`` column.
        cfg: Configuration dict with ``data.train_end`` and ``data.val_end``.

    Returns:
        A tuple ``(train_df, val_df, test_df)`` of DataFrames.
    """
    train_mask, val_mask, test_mask = get_split_masks(df, cfg)

    train = df[train_mask].copy()
    val = df[val_mask].copy()
    test = df[test_mask].copy()

    print(f"Train: {len(train):,} rows ({train['timestamp'].min()} to {train['timestamp'].max()})")
    print(f"Val:   {len(val):,} rows ({val['timestamp'].min()} to {val['timestamp'].max()})")
    print(f"Test:  {len(test):,} rows ({test['timestamp'].min()} to {test['timestamp'].max()})")

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
        train: Training split DataFrame.
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
    print(f"Saved features: {processed_path_obj} ({len(full):,} rows)")

    # Save train split as reference for drift detection
    reference_path_obj = Path(reference_path)
    reference_path_obj.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(reference_path_obj, index=False)
    print(f"Saved reference: {reference_path_obj} ({len(train):,} rows)")


def main():
    """Orchestrate feature engineering pipeline."""
    cfg = load_config()
    raw_path = Path(cfg["data"]["raw_path"]) / "synthetic.csv"
    processed_path = cfg["data"]["processed_path"]
    reference_path = cfg["data"]["reference_path"]

    print("=== Loading raw data ===")
    df = load_raw_data(str(raw_path))
    print(f"Loaded {len(df):,} rows from {raw_path}")

    # Sort by timestamp to ensure correct lag computation
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Calendar features
    if cfg["features"]["calendar"]["enabled"]:
        print("=== Adding calendar features ===")
        df = add_calendar_features(df)
        print(f"  Added: hour, day_of_week, month, week_of_year, is_holiday, is_workday")

    # Lag features
    if cfg["features"]["lags"]["enabled"]:
        periods = cfg["features"]["lags"]["periods"]
        print(f"=== Adding lag features: {periods} ===")
        df = add_lag_features(df, periods)
        print(f"  Added lag features for periods: {periods}")

    # Drop rows with NaN (from lag creation at start of series)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    if before - after > 0:
        print(f"Dropped {before - after} rows with NaN from lag creation")

    print("=== Splitting into train/val/test ===")
    train, val, test = train_val_test_split(df, cfg)

    print("=== Saving processed data ===")
    save_processed_data(train, val, test, processed_path, reference_path)
    print("Done.")


if __name__ == "__main__":
    main()
