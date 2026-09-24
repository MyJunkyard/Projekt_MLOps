"""
ingestion/weather/merge.py — UTC-safe weather merge into the price frame.

Alignment layer between acquisition and featurisation (Workstream 3,
decisions D5/D6/D9/D10): normalizes both join keys to tz-aware UTC,
renames weather columns to the ``{location}__{variable}`` contract, and
left-joins one location's weather onto the price frame.
"""

import logging

import numpy as np
import pandas as pd

from src.ingestion.weather.naming import (
    select_weather_columns,
    weather_feature_name,
)

# Stable module name (not `__name__` — under `python -m` it is `"__main__"`
# and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.ingestion.weather.merge"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def _as_utc_aware(series: pd.Series) -> pd.Series:
    """Normalize a datetime Series to tz-aware UTC.

    Both the featurise input frame (parsed from CSV, tz-aware UTC) and the
    weather frames (tz-aware UTC internally) are normalized to a single
    representation so the join key is unambiguous (decision D5, revised).
    Naive values are assumed to already be UTC (project convention — no DST
    handling) and are localized to UTC. The output is always tz-aware UTC,
    consistent with the rest of the pipeline (``get_split_masks`` in
    ``common/splits.py`` requires tz-aware timestamps) and with the schema
    declaration in ``features.main._base_specs``
    (``datetime64[ns, UTC]``).
    """
    return pd.to_datetime(series, utc=True)


def merge_weather(
    price_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    location: str,
) -> pd.DataFrame:
    """Left-join one location's weather onto the price frame by UTC time.

    Weather columns are renamed to ``{location}__{variable}`` (decision
    D9). Missing weather hours produce NaN rows (count INFO-logged) that
    the featurise stage's ``dropna()`` later removes (decision D6). If a
    ``wind_mw`` column is present (Workstream 4+), the
    ``wind_speed_100m``/``wind_mw`` correlation is logged as a
    timezone/merge sanity check (decision D10).

    Args:
        price_df: Price/load frame with a ``timestamp`` column (tz-aware
            or tz-naive UTC — normalized internally). Not mutated.
        weather_df: One location's weather frame (``timestamp`` + float
            variable columns). Not mutated.
        location: Location key used for the column suffix and messages.

    Returns:
        A new DataFrame: ``price_df`` columns plus
        ``{location}__{variable}`` columns for every weather variable.

    Raises:
        ValueError: If the weather frame is empty or contains duplicate
            timestamps, or the merge would duplicate rows.
    """
    if weather_df.empty:
        raise ValueError(
            f"Weather frame for {location!r} is empty — nothing to merge."
        )

    weather = weather_df.copy()
    weather["timestamp"] = _as_utc_aware(weather["timestamp"])
    if weather["timestamp"].duplicated().any():
        duplicates = weather.loc[
            weather["timestamp"].duplicated(), "timestamp"
        ].head(3).tolist()
        raise ValueError(
            f"Weather frame for {location!r} contains duplicate timestamps "
            f"(e.g. {duplicates}) — refusing to merge ambiguous data."
        )
    weather = weather.rename(
        columns={
            column: weather_feature_name(column, location)
            for column in weather.columns
            if column != "timestamp"
        }
    )
    weather_columns = [c for c in weather.columns if c != "timestamp"]

    price = price_df.copy()
    price["timestamp"] = _as_utc_aware(price["timestamp"])
    # many_to_one: each price hour matches at most one weather hour — a
    # final guard against duplicated weather keys corrupting the frame.
    merged = price.merge(
        weather, on="timestamp", how="left", validate="many_to_one"
    )

    missing_mask = merged[weather_columns].isna().any(axis=1)
    n_missing = int(missing_mask.sum())
    if n_missing:
        logger.info(
            "%d of %d rows have missing weather for %s (NaN policy: these "
            "rows are dropped by the featurise dropna gate)",
            n_missing,
            len(merged),
            location,
        )

    # Sanity check (decision D10): meaningful wind correlation is expected
    # once real generation data (Workstream 4) is present; near-zero means
    # the weather merge is likely misaligned.
    wind_columns = select_weather_columns(merged, variable="wind_speed_100m")
    if "wind_mw" in merged.columns and wind_columns:
        correlation = merged[wind_columns[0]].corr(merged["wind_mw"])
        logger.info(
            "Sanity check: corr(%s, wind_mw) = %.3f",
            wind_columns[0],
            correlation,
        )
        if np.isnan(correlation) or abs(correlation) < 0.05:
            logger.warning(
                "Near-zero wind speed / wind generation correlation — the "
                "weather merge may be misaligned (check timestamps/timezone)."
            )

    return merged
