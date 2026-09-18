"""
ingestion/validation.py — Schema checks, grid reindexing, and gap filling.

Extracted verbatim from ``ingest.py`` (Workstream 0 module restructure).
"""

import logging

import pandas as pd

from src.config.models import DataConfig, TemporalConfig

MODULE_LOGGER_NAME = "src.ingestion.validation"
logger = logging.getLogger(MODULE_LOGGER_NAME)

# Plausible price range for European day-ahead electricity prices (EUR/MWh).
# Negative prices occur (e.g. wind surplus); extreme outliers indicate errors.
PRICE_MIN = -500.0
PRICE_MAX = 500.0


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


def validate_entsoe_data(
    df: pd.DataFrame, data: DataConfig, temporal: TemporalConfig
) -> bool:
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
        data: ``DataConfig``; ``max_gap_periods`` sets the gap threshold
            (in periods of the temporal resolution).
        temporal: ``TemporalConfig``; ``resolution`` sets the expected
            data frequency.

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

    # Gap detection threshold from config. Expected period duration comes
    # from the temporal resolution.
    max_gap_periods = data.max_gap_periods
    resolution = temporal.resolution
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
