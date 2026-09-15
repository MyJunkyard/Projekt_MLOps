"""
ingestion/entsoe.py — ENTSO-E data acquisition.

Downloads day-ahead prices and actual load for the Polish (PSE) bidding
zone from the ENTSO-E Transparency Platform via the entsoe-py client.
Falls back to synthetic data generation when no API key is available.

Extracted verbatim from ``ingest.py`` (Workstream 0 module restructure).
"""

import logging
import os

import numpy as np
import pandas as pd

# ENTSoE client: entsoe-py >= 0.10 renamed the client to `EntsoeClient`;
# earlier versions used `EntsoePandasClient`. Support both.
try:
    from entsoe import EntsoeClient
except ImportError:
    from entsoe import EntsoePandasClient as EntsoeClient  # type: ignore

# Stable module name (not `__name__` — under `python -m` it is `"__main__"`
# and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.ingestion.entsoe"
logger = logging.getLogger(MODULE_LOGGER_NAME)


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
