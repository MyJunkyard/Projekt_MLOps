"""
ingest.py — Stage 2: Download real ENTSO-E data with validation and manifest.

Downloads day-ahead prices and actual load for the Polish (PSE) bidding zone
from the ENTSO-E Transparency Platform via the entsoe-py client. Validates
schema, reindexes to a complete hourly grid, forward-fills short gaps, and
writes a data manifest. Falls back to synthetic data when no API key is
available.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, setup_logging

# ENTSoE client: entsoe-py >= 0.10 renamed the client to `EntsoeClient`;
# earlier versions used `EntsoePandasClient`. Support both.
try:
    from entsoe import EntsoeClient
except ImportError:
    from entsoe import EntsoePandasClient as EntsoeClient  # type: ignore

# Stable module name (not `__name__` — under `python -m` it is `"__main__"`
# and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.ingest"
logger = logging.getLogger(MODULE_LOGGER_NAME)

# Plausible price range for European day-ahead electricity prices (EUR/MWh).
# Negative prices occur (e.g. wind surplus); extreme outliers indicate errors.
PRICE_MIN = -500.0
PRICE_MAX = 500.0


def _normalize_load_series(
    load: pd.Series | pd.DataFrame,
) -> pd.Series:
    """Normalize the entsoe-py ``query_load`` return value to a Series.

    Explicit boundary adapter: entsoe-py returns either a Series (older
    versions) or a single-column DataFrame with an ``"Actual Load"``
    header (newer versions). This function converts either shape to a
    Series so downstream code has one documented input contract.

    Args:
        load: Raw ``query_load`` result — a Series of load values or a
            DataFrame whose first column holds them.

    Returns:
        The load values as a ``pd.Series`` (unnamed; the caller renames
        it to ``load_mw``).
    """
    if isinstance(load, pd.DataFrame):
        return load.iloc[:, 0]
    return load


def generate_synthetic_data(
    n_hours: int = 50000, seed: int = 42, include_load: bool = True
) -> pd.DataFrame:
    """Generate synthetic hourly electricity price data.

    Price = base (50) + daily seasonality (sine, amplitude 10)
            + weekly seasonality (sine, amplitude 5) + noise (std 5)
    With occasional price spikes (×3 for 0.5% of rows)
    and occasional negative prices (0.1% of rows, realistic for electricity).

    Args:
        n_hours: Number of hourly timestamps to generate.
        seed: Random seed for reproducibility.
        include_load: Whether to also generate a synthetic ``load_mw`` column.

    Returns:
        A DataFrame with columns ``timestamp`` (tz-aware UTC) and
        ``price_eur_mwh`` (float), plus ``load_mw`` (float) when
        ``include_load`` is true.
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

    if include_load:
        # Synthetic actual load (MW): strong daily/weekly seasonality plus
        # noise, loosely correlated with the price seasonality. Keeps the
        # synthetic fallback schema-compatible with real ENTSO-E downloads
        # made with data.entsoe.include_load: true.
        load = (
            10000.0
            + 3000.0 * np.sin(2 * np.pi * time_steps / 24)
            + 1500.0 * np.sin(2 * np.pi * time_steps / (24 * 7))
            + rng.normal(0, 500.0, size=n_hours)
        )
        df["load_mw"] = load

    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame has the expected schema.

    Required DataFrame contract:
        - ``timestamp``: datetime column, no nulls, monotonically
          increasing, no duplicates.
        - ``price_eur_mwh``: numeric column, no nulls.

    Checks:
    - timestamp column exists and is datetime
    - price_eur_mwh column exists and is numeric
    - No nulls in either column
    - Timestamps are monotonically increasing
    - No duplicate timestamps

    Args:
        df: DataFrame to validate (contract above).

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


def download_entsoe_data(cfg: dict) -> pd.DataFrame:
    """Download day-ahead prices and actual load from ENTSO-E.

    Uses the entsoe-py client to fetch data for the configured bidding zone
    (default: PSE / Poland). Reads the API key from the ``ENTSOE_API_KEY``
    environment variable.

    Args:
        cfg: Configuration dict with ``data.entsoe.bidding_zone``,
            ``data.entsoe.start_date`` and optionally
            ``data.entsoe.include_load``.

    Returns:
        A DataFrame with ``timestamp`` (tz-aware UTC) and ``price_eur_mwh``
        (float) columns, plus ``load_mw`` (float) when
        ``data.entsoe.include_load`` is true (the default).

    Raises:
        ValueError: If no ENTSOE_API_KEY environment variable is set.
    """
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise ValueError(
            "ENTSOE_API_KEY environment variable is not set. "
            "Falling back to synthetic data."
        )

    entsoe_cfg = cfg.get("data", {}).get("entsoe", {})
    bidding_zone = entsoe_cfg.get("bidding_zone", "PSE")
    start_date = entsoe_cfg.get("start_date", "2018-01-01")

    client = EntsoeClient(api_key=api_key)

    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp.now(tz="UTC")

    logger.info(
        "Downloading ENTSO-E data for bidding zone '%s' from %s to %s",
        bidding_zone,
        start,
        end,
    )

    # Download day-ahead prices
    prices = client.query_day_ahead_prices(
        bidding_zone, start=start, end=end
    )

    df = pd.DataFrame({"timestamp": prices.index, "price_eur_mwh": prices.values})

    # Download actual load and merge it as a load_mw column (review point 3:
    # the result is no longer discarded). Gated by data.entsoe.include_load.
    include_load = entsoe_cfg.get("include_load", True)
    if include_load:
        load = _normalize_load_series(
            client.query_load(bidding_zone, start=start, end=end)
        )
        load = load.rename("load_mw")
        load_df = load.to_frame()
        load_df.index.name = "timestamp"
        # Outer join so wider load coverage doesn't silently drop hours;
        # any resulting NaN rows are handled by the gap-filling stage.
        df = df.merge(load_df.reset_index(), on="timestamp", how="outer")
        logger.info("Downloaded %d load records", len(load))

    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("Downloaded %d price records", len(df))
    return df


def validate_entsoe_data(df: pd.DataFrame, cfg: dict) -> bool:
    """Validate ENTSO-E data for gaps, outliers, and timezone correctness.

    Required DataFrame contract:
        - ``timestamp``: tz-aware UTC datetime column, no duplicates.
        - ``price_eur_mwh``: numeric column, no nulls.
        - ``load_mw`` (optional): numeric column when present; NaNs
          allowed (gaps are handled by the gap-filling stage).

    Checks:
    - Required columns present (timestamp, price_eur_mwh)
    - ``load_mw`` column, when present (``data.entsoe.include_load``), is
      numeric (NaNs allowed — gaps are handled by the gap-filling stage)
    - Timestamp is tz-aware UTC
    - No duplicate timestamps
    - Gaps larger than the configured ``data.max_gap_periods`` are *warned*
      about, not raised on — the gap-filling stage (``fill_gaps``) is the
      component responsible for handling them. Raising here would make the
      long-gap warn path in ``fill_gaps`` dead code for real data.
    - Prices within plausible range (PRICE_MIN to PRICE_MAX)

    Args:
        df: DataFrame to validate (contract above).
        cfg: Configuration dict. ``data.max_gap_periods`` (default 2) sets the
            gap threshold (in periods of ``temporal.resolution``);
            ``temporal.resolution`` (default ``"hourly"``)
            sets the expected data frequency.

    Returns:
        True if all checks pass.

    Raises:
        ValueError: If any validation check fails.
    """
    # Column existence
    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")
    if "price_eur_mwh" not in df.columns:
        raise ValueError("Missing required column: price_eur_mwh")

    # Optional load column (present when data.entsoe.include_load is true).
    # NaNs are allowed here — gaps are handled by the gap-filling stage.
    if "load_mw" in df.columns and not pd.api.types.is_numeric_dtype(
        df["load_mw"]
    ):
        raise ValueError("load_mw column must be numeric type")

    # Timezone check
    if df["timestamp"].dt.tz is None:
        raise ValueError("timestamp column must be timezone-aware (UTC)")
    if str(df["timestamp"].dt.tz) != "UTC":
        raise ValueError("timestamp column must be in UTC timezone")

    # Duplicate timestamps
    if df["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps found in ENTSO-E data")

    # Sort for gap detection
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    # Gap detection threshold derived from config; falls back to 2 if unset.
    # Expected period duration comes from temporal.resolution.
    max_gap_periods = cfg.get("data", {}).get("max_gap_periods", 2)
    resolution = cfg.get("temporal", {}).get("resolution", "hourly")
    period_map = {
        "hourly": pd.Timedelta(hours=1),
        "daily": pd.Timedelta(days=1),
        "weekly": pd.Timedelta(weeks=1),
    }
    period = period_map.get(resolution, pd.Timedelta(hours=1))

    time_diffs = df_sorted["timestamp"].diff().dropna()
    max_gap = time_diffs.max() if len(time_diffs) > 0 else pd.Timedelta(0)
    # A gap of k missing periods appears as a timestamp diff of (k+1) periods.
    # Warn on gaps that fill_gaps will NOT impute: k > max_gap_periods, i.e.
    # diff > (max_gap_periods + 1) * period. This keeps the validation and the
    # imputation stage consistent (review points 7 and 13).
    gap_threshold = (max_gap_periods + 1) * period
    gap_count = (time_diffs > gap_threshold).sum()
    if gap_count > 0:
        logger.warning(
            "Found %d gap(s) larger than %d %s period(s) in ENTSO-E data "
            "(resolution=%s). Maximum gap: %s. These will be handled by "
            "the gap-filling stage; check the manifest for imputation stats.",
            gap_count,
            max_gap_periods,
            resolution,
            resolution,
            max_gap,
        )

    # Outlier detection
    prices = df_sorted["price_eur_mwh"]
    if prices.min() < PRICE_MIN or prices.max() > PRICE_MAX:
        raise ValueError(
            f"Price outliers detected: min={prices.min():.2f}, "
            f"max={prices.max():.2f}. Expected range: "
            f"[{PRICE_MIN}, {PRICE_MAX}]"
        )

    return True


def reindex_to_grid(
    df: pd.DataFrame, freq: str = "h"
) -> pd.DataFrame:
    """Reindex a DataFrame to a complete, regular time grid.

    Builds a continuous grid from the min to max timestamp at the given
    frequency and reindexes the data onto it, inserting NaN rows for any
    missing periods. This is the foundation for gap detection and
    imputation: after reindexing, a gap of ``k`` missing periods appears as
    ``k`` consecutive NaN rows.

    Args:
        df: DataFrame with a tz-aware UTC ``timestamp`` column and at
            least one data column.
        freq: Pandas frequency string for the grid (e.g. ``"h"``, ``"D"``).

    Returns:
        A DataFrame reindexed to a complete regular grid, with the same
        columns as the input. Missing periods are filled with NaN.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Build complete grid from min to max timestamp
    full_range = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq=freq,
        tz="UTC",
    )

    # Reindex to fill missing periods
    df_indexed = df.set_index("timestamp")
    df_reindexed = df_indexed.reindex(full_range)
    df_reindexed = df_reindexed.reset_index().rename(
        columns={"index": "timestamp"}
    )

    return df_reindexed


def fill_gaps(
    df: pd.DataFrame,
    max_gap_periods: int = 2,
    fill_method: str = "ffill",
    freq: str = "h",
    add_is_imputed_flag: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Impute short gaps by reindexing to a grid and filling only short runs.

    This fixes the previous behaviour where ``df.ffill()`` was applied
    unconditionally, which forward-filled gaps of *any* size. Now:

    1. The data is reindexed to a complete grid of the given frequency.
    2. Contiguous runs of missing values are identified.
    3. Only runs of at most ``max_gap_periods`` periods are imputed using the
       configured ``fill_method``. Longer runs remain NaN (they are dropped
       downstream, never silently filled with stale values).
    4. If ``add_is_imputed_flag``, an ``is_imputed`` boolean column is added
       marking which rows were fabricated by imputation. This makes the
       imputed-vs-real distinction visible and auditable downstream.

    ``fill_method`` is pluggable so the imputation strategy can be swapped
    (e.g. ``"ffill"`` vs ``"interpolate"``) without changing the callers.
    This matters for later time-derivative features, where linear
    interpolation of a smooth signal is more principled than carrying the
    last value forward.

    Args:
        df: DataFrame with a tz-aware UTC ``timestamp`` column and at
            least one data column (e.g. ``price_eur_mwh``, ``load_mw``).
        max_gap_periods: Maximum number of consecutive missing periods to impute.
        fill_method: Imputation strategy. Supports ``"ffill"`` (forward-fill)
            and ``"interpolate"`` (linear interpolation). Extensible to any
            pandas ``fillna``/``interpolate`` keyword.
        freq: Frequency of the grid used to detect gaps (``"h"`` for hourly).
        add_is_imputed_flag: Whether to add the ``is_imputed`` column.

    Returns:
        A tuple ``(df, stats)`` where ``df`` has short gaps imputed and longer
        gaps left as NaN (plus an ``is_imputed`` boolean column when
        ``add_is_imputed_flag`` is true), and ``stats`` is a dict with keys
        ``n_imputed``, ``n_unfilled``, ``max_gap_periods``, ``fill_method``,
        and ``freq`` for recording in the manifest.
    """
    if fill_method not in {"ffill", "interpolate"}:
        raise ValueError(
            f"Unsupported fill_method: {fill_method!r}. "
            f"Supported: 'ffill', 'interpolate'."
        )

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = reindex_to_grid(df, freq=freq)

    data_cols = [c for c in df.columns if c != "timestamp"]
    if not data_cols:
        raise ValueError("DataFrame has no data columns to fill")

    # Boolean mask of rows that are missing data (any data column is NaN)
    missing_mask = df[data_cols].isna().any(axis=1)
    # Identify contiguous runs of missing rows
    run_ids = (missing_mask & ~missing_mask.shift(fill_value=False)).cumsum()
    # Length of each contiguous missing run, broadcast to every row in the run
    run_lengths = missing_mask.groupby(run_ids).transform("sum")

    # Only runs that are short enough are eligible for imputation. Filling is
    # applied to the whole frame (so isolated NaN rows see their non-NaN
    # neighbours), then rows in long runs are blanked back out to NaN.
    fillable = missing_mask & (run_lengths <= max_gap_periods)

    if fill_method == "ffill":
        filled = df.ffill()
    else:  # interpolate
        filled = df.interpolate(method="linear")

    for col in data_cols:
        col_missing = df[col].isna()
        # Keep the filled value only where we are allowed to impute;
        # non-fillable missing rows (long gaps) revert to NaN.
        df[col] = filled[col].where(~col_missing | fillable)

    # A row counts as imputed if it was fillable AND actually received a value
    # (e.g. leading NaN rows have nothing to forward-fill from).
    imputed_mask = fillable & ~df[data_cols].isna().any(axis=1)
    n_imputed = int(imputed_mask.sum())

    # Rows still NaN after imputation are the "long gaps" we refuse to fill
    n_unfilled = int(df[data_cols].isna().any(axis=1).sum())

    if n_imputed > 0:
        logger.info(
            "Imputed %d row(s) using %s (max_gap_periods=%d)",
            n_imputed,
            fill_method,
            max_gap_periods,
        )
    if n_unfilled > 0:
        logger.warning(
            "%d row(s) in gaps longer than max_gap_periods=%d were not imputed "
            "and remain as NaN (to be dropped downstream).",
            n_unfilled,
            max_gap_periods,
        )

    if add_is_imputed_flag:
        df["is_imputed"] = imputed_mask

    stats = {
        "n_imputed": n_imputed,
        "n_unfilled": n_unfilled,
        "max_gap_periods": max_gap_periods,
        "fill_method": fill_method,
        "freq": freq,
    }
    return df, stats


def forward_fill_gaps(
    df: pd.DataFrame, max_gap_periods: int = 2
) -> pd.DataFrame:
    """Backward-compatible wrapper around ``fill_gaps``.

    Forward-fills gaps up to ``max_gap_periods``; longer gaps remain as NaN.
    Kept so existing callers/tests that only need the filled DataFrame keep
    working. Prefer ``fill_gaps`` for new code — it returns imputation
    statistics and supports pluggable fill methods.

    Args:
        df: DataFrame with a ``timestamp`` column and at least one data column.
        max_gap_periods: Maximum gap size (in periods) to forward-fill.

    Returns:
        A DataFrame with short gaps forward-filled. Longer gaps remain as NaN.
    """
    filled, _ = fill_gaps(
        df,
        max_gap_periods=max_gap_periods,
        fill_method="ffill",
        add_is_imputed_flag=False,
    )
    return filled


def write_manifest(
    raw_path: str,
    df: pd.DataFrame,
    sha256_hash: str | None = None,
    imputation_stats: dict | None = None,
) -> None:
    """Write a data manifest JSON file with download metadata.

    The manifest records the integrity hash, row counts, and — when
    ``imputation_stats`` is provided — how many rows were imputed vs.
    dropped, so that every run's data-quality story is auditable
    (see review point 1b). If ``sha256_hash`` is supplied it is used
    directly (this is the on-disk file hash from ``save_raw_data``),
    so the manifest and the printed summary reference the same file.

    Args:
        raw_path: Path to the raw data directory.
        df: The raw DataFrame that was saved. Must have a ``timestamp``
            column (used for the ``date_range`` manifest entry).
        sha256_hash: SHA256 of the saved CSV file. Computed from the
            DataFrame if not provided.
        imputation_stats: Optional dict with imputation statistics
            (``n_imputed``, ``n_unfilled``, ``max_gap_periods``,
            ``fill_method``, ``freq``).
    """
    path_obj = Path(raw_path)
    path_obj.mkdir(parents=True, exist_ok=True)

    if sha256_hash is None:
        # Compute SHA256 of the DataFrame content
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        sha256_hash = hashlib.sha256(csv_bytes).hexdigest()

    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "date_range": [
            df["timestamp"].min().isoformat(),
            df["timestamp"].max().isoformat(),
        ],
        "row_count": len(df),
        "sha256": sha256_hash,
    }

    if imputation_stats:
        manifest["n_imputed_rows"] = imputation_stats.get("n_imputed", 0)
        manifest["n_dropped_rows"] = imputation_stats.get("n_dropped_rows", 0)
        # fill_gaps reports "n_unfilled"; main() may normalize it to
        # "n_unfilled_rows" after deciding what dropna() removes.
        manifest["n_unfilled_rows"] = imputation_stats.get(
            "n_unfilled_rows",
            imputation_stats.get("n_unfilled", 0),
        )
        manifest["max_gap_periods"] = imputation_stats.get("max_gap_periods", 2)
        manifest["fill_method"] = imputation_stats.get("fill_method", "ffill")
        manifest["freq"] = imputation_stats.get("freq", "h")

    manifest_path = path_obj / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written to %s", manifest_path)


def save_raw_data(df: pd.DataFrame, path: str) -> str:
    """Save raw data to CSV, log a summary, and return the on-disk SHA256.

    Args:
        df: DataFrame to save. Must have a ``timestamp`` column (used
            for the logged data-range summary).
        path: Target CSV path.

    Returns:
        The SHA256 hex digest of the on-disk file. Callers should pass this
        to ``write_manifest`` so the manifest and the logged summary come
        from the same source (fixes review point 9).
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path_obj, index=False)

    # Compute SHA256 of the on-disk file
    sha256_hash = hashlib.sha256(path_obj.read_bytes()).hexdigest()

    logger.info("Saved %s", path_obj)
    logger.info("Saved %s rows", f"{len(df):,}")
    logger.info(
        "Data range: %s to %s",
        df["timestamp"].min(),
        df["timestamp"].max(),
    )
    logger.info("SHA256: %s", sha256_hash)
    return sha256_hash


def main():
    """Orchestrate: download → validate → fill → drop → save → manifest."""
    cfg = load_config()
    setup_logging(cfg, logger_name=MODULE_LOGGER_NAME)
    raw_path = cfg["data"]["raw_path"]
    max_gap_periods = cfg.get("data", {}).get("max_gap_periods", 2)
    fill_method = cfg.get("data", {}).get("fill_method", "ffill")
    freq = {
        "hourly": "h",
        "daily": "D",
        "weekly": "W",
    }.get(cfg.get("temporal", {}).get("resolution", "hourly"), "h")

    logger.info("Stage: ingestion")

    # Try to download real data; fall back to synthetic if no API key
    try:
        logger.info("Attempting ENTSO-E data download")
        df = download_entsoe_data(cfg)
        logger.info("Downloaded %s rows from ENTSO-E", f"{len(df):,}")
    except ValueError as e:
        logger.warning("ENTSO-E download unavailable: %s", e)
        logger.warning("Falling back to synthetic data generation")
        include_load = (
            cfg.get("data", {}).get("entsoe", {}).get("include_load", True)
        )
        df = generate_synthetic_data(include_load=include_load)
        logger.info("Generated %s synthetic rows", f"{len(df):,}")

    # Guard against empty data (review point 4): a successful download can
    # still return zero rows (e.g. no data in the requested date range),
    # which would otherwise crash later in reindex_to_grid with an opaque
    # ValueError from df["timestamp"].min() on an empty series.
    if df.empty:
        logger.error(
            "Ingestion produced an empty DataFrame (no rows in the requested "
            "date range)"
        )
        raise ValueError(
            "Ingestion produced an empty DataFrame (no rows in the requested "
            "date range). Check data.entsoe.start_date / bidding_zone or "
            "API availability."
        )

    # Validate
    logger.debug("Data columns: %s", list(df.columns))
    logger.info("Validating data")
    if "entsoe" in cfg.get("data", {}):
        validate_entsoe_data(df, cfg)
    else:
        validate_schema(df)
    logger.info("Validation passed")

    # Fill short gaps only (long gaps remain NaN — never ffilled).
    # fill_gaps reindexes to the config-derived grid internally (review point
    # 7: no hardcoded hourly assumption in main() — with temporal.resolution
    # set to daily/weekly, reindexing here at "h" would fabricate 23/167 NaN
    # rows per period that fill_gaps would then misclassify as long gaps).
    logger.info(
        "Filling gaps (max_gap_periods=%d, fill_method=%s, freq=%s)",
        max_gap_periods,
        fill_method,
        freq,
    )
    df, imputation_stats = fill_gaps(
        df,
        max_gap_periods=max_gap_periods,
        fill_method=fill_method,
        freq=freq,
    )

    # Explicitly drop (or retain) rows that remain NaN after imputation —
    # governed by the data.drop_long_gaps config flag (review points 1 and 1b).
    n_unfilled = imputation_stats["n_unfilled"]
    if cfg.get("data", {}).get("drop_long_gaps", True):
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        n_dropped = before - len(df)
        if n_dropped > 0:
            logger.warning(
                "Dropped %d rows with remaining NaN values "
                "(gaps longer than max_gap_periods=%d)",
                n_dropped,
                max_gap_periods,
            )
        # All unfilled rows were removed, so nothing remains unfilled
        imputation_stats["n_unfilled_rows"] = 0
    else:
        n_dropped = 0
        imputation_stats["n_unfilled_rows"] = n_unfilled
        if n_unfilled > 0:
            logger.info(
                "Retaining %d unfilled row(s) (drop_long_gaps=false); "
                "downstream stages may drop them.",
                n_unfilled,
            )
    imputation_stats["n_dropped_rows"] = n_dropped

    # Save
    output_path = Path(raw_path) / "entsoe_prices.csv"
    csv_hash = save_raw_data(df, str(output_path))

    # Write manifest — includes imputation/drop stats (review point 1b)
    write_manifest(
        raw_path, df, sha256_hash=csv_hash, imputation_stats=imputation_stats
    )

    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()
