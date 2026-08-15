"""
ingest.py — Stage 1: Generate synthetic data mimicking hourly electricity prices.

Generates a CSV with timestamp and price_eur_mwh columns, validates schema,
and saves to data/raw/. This is a stub — replace with ENTSO-E download in Stage 2.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config


def generate_synthetic_data(
    n_hours: int = 50000, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic hourly electricity price data.

    Price = base (50) + daily seasonality (sine, amplitude 10)
            + weekly seasonality (sine, amplitude 5) + noise (std 5)
    With occasional price spikes (×3 for 0.5% of rows)
    and occasional negative prices (0.1% of rows, realistic for electricity).

    Args:
        n_hours: Number of hourly timestamps to generate.
        seed: Random seed for reproducibility.

    Returns:
        A DataFrame with columns ``timestamp`` (tz-aware UTC) and
        ``price_eur_mwh`` (float).
    """
    rng = np.random.default_rng(seed)

    # Hourly timestamps starting from 2020-01-01
    start = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
    timestamps = pd.date_range(start=start, periods=n_hours, freq="h")

    # Base price components
    time_steps = np.arange(n_hours, dtype=float)
    base = 50.0
    daily_seasonality = 10.0 * np.sin(2 * np.pi * time_steps / 24)
    weekly_seasonality = 5.0 * np.sin(2 * np.pi * time_steps / (24 * 7))
    noise = rng.normal(0, 5.0, size=n_hours)

    price = base + daily_seasonality + weekly_seasonality + noise

    # Price spikes (multiply by 3 for 0.5% of rows)
    spike_mask = rng.random(n_hours) < 0.005
    price[spike_mask] *= 3.0

    # Negative prices (flip sign for 0.1% of rows)
    negative_mask = rng.random(n_hours) < 0.001
    price[negative_mask] = -np.abs(price[negative_mask])

    df = pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": price})
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame has the expected schema.

    Checks:
    - timestamp column exists and is datetime
    - price_eur_mwh column exists and is numeric
    - No nulls in either column
    - Timestamps are monotonically increasing
    - No duplicate timestamps

    Args:
        df: DataFrame to validate.

    Returns:
        True if all checks pass.

    Raises:
        ValueError: If any schema check fails.
    """
    # Column existence
    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")
    if "price_eur_mwh" not in df.columns:
        raise ValueError("Missing required column: price_eur_mwh")

    # Column types
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError("timestamp column must be datetime type")

    if not pd.api.types.is_numeric_dtype(df["price_eur_mwh"]):
        raise ValueError("price_eur_mwh column must be numeric type")

    # Null checks
    if df["timestamp"].isnull().any():
        raise ValueError("timestamp column contains null values")
    if df["price_eur_mwh"].isnull().any():
        raise ValueError("price_eur_mwh column contains null values")

    # Monotonically increasing timestamps
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps are not monotonically increasing")

    # Duplicate timestamps
    if df["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps found")

    return True


def save_raw_data(df: pd.DataFrame, path: str) -> None:
    """Save raw data to CSV and print summary."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path_obj, index=False)

    # Compute SHA256 of the on-disk file
    sha256_hash = hashlib.sha256(path_obj.read_bytes()).hexdigest()

    print(f"Saved: {path_obj}")
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"SHA256: {sha256_hash}")


def main():
    """Orchestrate: generate → validate → save."""
    cfg = load_config()
    raw_path = cfg["data"]["raw_path"]

    print("=== Stage 1: Synthetic Data Generation ===")
    print("Generating synthetic hourly price data...")
    df = generate_synthetic_data()

    print("Validating schema...")
    validate_schema(df)
    print("Schema validation passed.")

    output_path = Path(raw_path) / "synthetic.csv"
    save_raw_data(df, str(output_path))
    print("Done.")
    # TODO: replace with ENTSO-E download in Stage 2


if __name__ == "__main__":
    main()