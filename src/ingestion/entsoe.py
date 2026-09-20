"""
ingestion/entsoe.py — ENTSO-E data acquisition.

Downloads day-ahead prices, actual load, and actual generation per type
for the Polish (PSE) bidding zone from the ENTSO-E Transparency Platform
via the entsoe-py client. Falls back to synthetic data generation when
no API key is available.

Workstream 4 (generation mix) — deployment-leakage note: the actual
generation per type is only published with a real-time lag (~1h). The
raw ``{source}_mw`` columns stored here are therefore **aligned to
their availability** by the featurise stage (``features.availability_lags``
→ ``{source}_mw_lag{L}h``) before they can become model features; the
raw current-hour values never reach ``features.parquet``. The ENTSO-E
day-ahead generation *forecast* is the Stage 4 alternative (EXP-005)
for a like-for-like comparison of forecast- vs. lagged-actual features.
"""

import json
import logging
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.models import EntsoeConfig, GenerationMixConfig
from src.ingestion.manifest import SOURCE_ENTSOE, SOURCE_SYNTHETIC, save_raw_data

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

#: Lowercase substrings used to map entsoe-py per-PSR-type column names
#: (e.g. ``"Wind Offshore"``, ``"Fossil Hard coal"``) to the configured
#: generation sources. A source matches a column when any pattern is a
#: substring of the lowercased column name; multiple matches are summed.
PSR_TYPE_PATTERNS: dict[str, tuple[str, ...]] = {
    "wind": ("wind",),
    "solar": ("solar",),
    "coal": ("hard coal", "brown coal", "lignite", "coal"),
    "gas": ("fossil gas", "gas"),
    "nuclear": ("nuclear",),
    "hydro": ("hydro",),
}

_GENERATION_CACHE_DIR_NAME = "entsoe/generation"
_GENERATION_CSV_NAME = "generation.csv"
_GENERATION_MANIFEST_NAME = "manifest.json"


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


def generate_synthetic_generation(
    timestamps: pd.Series | pd.DatetimeIndex,
    sources: Sequence[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic per-source generation columns for timestamps.

    Plausible shapes per source (schema-compatible fallback for offline
    runs and tests): solar follows a daylight half-sine (zero at night),
    wind is a slow multi-day sinusoid clipped at zero, coal/gas carry a
    smooth base with daily modulation, nuclear/hydro are near-constant.

    Args:
        timestamps: Hourly tz-aware UTC timestamps to generate values for.
        sources: Generation source names (keys of ``PSR_TYPE_PATTERNS``).
        seed: Random seed for reproducibility.

    Returns:
        A DataFrame indexed like ``timestamps`` (positional 0..n-1) with
        one float ``{source}_mw`` column per requested source.

    Raises:
        ValueError: If a source is not a known generation source.
    """
    unknown = sorted(set(sources) - set(PSR_TYPE_PATTERNS))
    if unknown:
        raise ValueError(
            f"Unknown generation source(s) {unknown} — known sources: "
            f"{sorted(PSR_TYPE_PATTERNS)}"
        )
    rng = np.random.default_rng(seed)
    n = len(timestamps)
    hours = np.asarray(pd.DatetimeIndex(timestamps).hour, dtype=float)
    time_steps = np.arange(n, dtype=float)
    daylight = np.clip(np.sin(np.pi * (hours - 6.0) / 12.0), 0.0, None)

    shapes: dict[str, np.ndarray] = {
        "solar": 1200.0 * daylight,
        "wind": np.clip(
            1500.0 + 1000.0 * np.sin(2 * np.pi * time_steps / 72.0)
            + rng.normal(0, 200.0, size=n),
            0.0,
            None,
        ),
        "coal": 8000.0 + 800.0 * np.sin(2 * np.pi * time_steps / 24.0)
        + rng.normal(0, 150.0, size=n),
        "gas": 3000.0 + 600.0 * np.sin(2 * np.pi * time_steps / 24.0 + 1.0)
        + rng.normal(0, 150.0, size=n),
        "nuclear": 7000.0 + rng.normal(0, 50.0, size=n),
        "hydro": 1000.0 + 200.0 * np.sin(2 * np.pi * time_steps / (24 * 7.0))
        + rng.normal(0, 50.0, size=n),
    }
    return pd.DataFrame(
        {f"{source}_mw": np.clip(shapes[source], 0.0, None) for source in sources},
        index=pd.DatetimeIndex(timestamps),
    )


def generate_synthetic_data(
    n_hours: int = 50000,
    seed: int = 42,
    include_load: bool = True,
    include_generation: bool = False,
    generation_sources: Sequence[str] | None = None,
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
        include_generation: Whether to also generate synthetic
            ``{source}_mw`` generation columns (Workstream 4) so the
            fallback stays schema-compatible with generation-mix runs.
        generation_sources: Sources to generate when
            ``include_generation`` is true; defaults to all known sources.

    Returns:
        A DataFrame with columns ``timestamp`` (tz-aware UTC) and
        ``price_eur_mwh`` (float), plus ``load_mw`` (float) when
        ``include_load`` is true and ``{source}_mw`` columns when
        ``include_generation`` is true.
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

    if include_generation:
        sources = (
            generation_sources
            if generation_sources is not None
            else sorted(PSR_TYPE_PATTERNS)
        )
        generation = generate_synthetic_generation(timestamps, sources, seed=seed)
        df = pd.concat(
            [df.reset_index(drop=True), generation.reset_index(drop=True)], axis=1
        )

    return df


def download_entsoe_data(entsoe: EntsoeConfig) -> pd.DataFrame:
    """Download day-ahead prices and actual load from ENTSO-E.

    Uses the entsoe-py client to fetch data for the configured bidding zone
    (default: PSE / Poland). Reads the API key from the ``ENTSOE_API_KEY``
    environment variable.

    Args:
        entsoe: ``EntsoeConfig`` with ``bidding_zone``, ``start_date``,
            and ``include_load``.

    Returns:
        A DataFrame with ``timestamp`` (tz-aware UTC) and ``price_eur_mwh``
        (float) columns, plus ``load_mw`` (float) when ``include_load``
        is true (the default).

    Raises:
        ValueError: If no ENTSOE_API_KEY environment variable is set.
    """
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise ValueError(
            "ENTSOE_API_KEY environment variable is not set. "
            "Falling back to synthetic data."
        )

    bidding_zone = entsoe.bidding_zone

    client = EntsoeClient(api_key=api_key)

    start = pd.Timestamp(entsoe.start_date, tz="UTC")
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
    # the result is no longer discarded). Gated by include_load.
    if entsoe.include_load:
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


# ---------------------------------------------------------------------------
# Generation mix (Workstream 4)
# ---------------------------------------------------------------------------


def map_generation_sources(
    gen_df: pd.DataFrame, sources: Sequence[str]
) -> pd.DataFrame:
    """Map entsoe-py per-PSR-type columns to ``{source}_mw`` columns.

    A source matches a column when any of its ``PSR_TYPE_PATTERNS``
    substrings occurs in the lowercased column name; multiple matching
    columns are summed per timestamp. A source with no matching column
    becomes an all-NaN column with a WARNING (documented policy: keep
    the column so the schema stays stable; the gap-filling stage and
    the manifest's per-source non-null counts make the coverage
    visible).

    Args:
        gen_df: entsoe-py ``query_generation`` result — a DataFrame
            indexed by tz-aware UTC timestamp with one column per PSR
            type (MultiIndex columns are flattened to their first level
            first).
        sources: Generation source names (keys of ``PSR_TYPE_PATTERNS``).

    Returns:
        A DataFrame indexed like ``gen_df`` with one float
        ``{source}_mw`` column per requested source.

    Raises:
        ValueError: If a source is not a known generation source.
    """
    unknown = sorted(set(sources) - set(PSR_TYPE_PATTERNS))
    if unknown:
        raise ValueError(
            f"Unknown generation source(s) {unknown} — known sources: "
            f"{sorted(PSR_TYPE_PATTERNS)}"
        )
    if isinstance(gen_df.columns, pd.MultiIndex):
        gen_df = gen_df.copy()
        gen_df.columns = gen_df.columns.get_level_values(0)

    out = pd.DataFrame(index=gen_df.index)
    for source in sources:
        patterns = PSR_TYPE_PATTERNS[source]
        matching = [
            col
            for col in gen_df.columns
            if any(pattern in str(col).lower() for pattern in patterns)
        ]
        if matching:
            out[f"{source}_mw"] = gen_df[matching].sum(axis=1)
        else:
            logger.warning(
                "No ENTSO-E generation column matches source %r "
                "(patterns: %s) — writing an all-NaN column",
                source,
                patterns,
            )
            out[f"{source}_mw"] = np.nan
    return out


def download_generation_mix(
    entsoe: EntsoeConfig,
    sources: Sequence[str],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Download actual generation per type from ENTSO-E (year-chunked).

    Queries ``query_generation`` for the configured bidding zone in
    **calendar-year chunks** (7 years of hourly per-type data in one
    call times out — plan risk #5), concatenates the chunks, and maps
    the per-PSR-type columns to ``{source}_mw`` columns via
    :func:`map_generation_sources`.

    Args:
        entsoe: ``EntsoeConfig`` with ``bidding_zone`` and ``start_date``.
        sources: Generation source names (keys of ``PSR_TYPE_PATTERNS``).
        start: Range start (defaults to ``entsoe.start_date``).
        end: Range end (defaults to now, UTC).

    Returns:
        A DataFrame with ``timestamp`` (tz-aware UTC) and one float
        ``{source}_mw`` column per requested source.

    Raises:
        ValueError: If no ENTSOE_API_KEY environment variable is set.
    """
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise ValueError(
            "ENTSOE_API_KEY environment variable is not set. "
            "Falling back to synthetic data."
        )
    start = pd.Timestamp(entsoe.start_date, tz="UTC") if start is None else start
    end = pd.Timestamp.now(tz="UTC") if end is None else end
    bidding_zone = entsoe.bidding_zone

    client = EntsoeClient(api_key=api_key)
    logger.info(
        "Downloading ENTSO-E generation mix for '%s' from %s to %s "
        "(year-chunked)",
        bidding_zone,
        start,
        end,
    )

    chunks: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, pd.Timestamp(f"{year}-01-01", tz="UTC"))
        chunk_end = min(end, pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        logger.debug(
            "Generation chunk %d: %s .. %s", year, chunk_start, chunk_end
        )
        chunk = client.query_generation(
            bidding_zone, start=chunk_start, end=chunk_end
        )
        chunks.append(chunk)
    gen_df = pd.concat(chunks).sort_index()
    gen_df = gen_df[~gen_df.index.duplicated(keep="last")]

    mapped = map_generation_sources(gen_df, sources)
    mapped = mapped.reset_index().rename(columns={"index": "timestamp"})
    logger.info("Downloaded %d generation records", len(mapped))
    return mapped


def _generation_cache_dir(raw_dir: str | Path) -> Path:
    """Return the generation cache directory under ``raw_dir``."""
    return Path(raw_dir) / _GENERATION_CACHE_DIR_NAME


def _read_generation_manifest(cache_dir: Path) -> dict:
    """Read the generation cache manifest (empty dict when absent)."""
    manifest_path = cache_dir / _GENERATION_MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def _generation_entry_covers(
    entry: dict, start: pd.Timestamp, end: pd.Timestamp, sources: Sequence[str]
) -> bool:
    """Check whether a generation manifest entry covers range + sources."""
    try:
        cached_start = pd.Timestamp(entry["date_range"][0])
        cached_end = pd.Timestamp(entry["date_range"][1])
        return bool(
            cached_start <= pd.Timestamp(start)
            and cached_end >= pd.Timestamp(end)
            and set(sources) <= set(entry["generation_sources"])
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def _write_generation_cache(
    cache_dir: Path,
    df: pd.DataFrame,
    source: str,
    sources: Sequence[str],
) -> None:
    """Persist the generation frame + manifest entry (weather-cache style)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / _GENERATION_CSV_NAME
    csv_hash = save_raw_data(df, str(csv_path))
    manifest = {
        "source": source,
        "sha256": csv_hash,
        "date_range": [
            df["timestamp"].min().isoformat(),
            df["timestamp"].max().isoformat(),
        ],
        "row_count": len(df),
        "generation_sources": sources,
        # Per-source non-null row counts make per-source coverage visible
        # (ENTSO-E per-type data has gaps and missing PSR types).
        "non_null_rows": {
            col: int(df[col].notna().sum())
            for col in df.columns
            if col != "timestamp"
        },
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(cache_dir / _GENERATION_MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump({"generation": manifest}, f, indent=2)
    logger.info(
        "Generation manifest written to %s", cache_dir / _GENERATION_MANIFEST_NAME
    )


def ingest_generation_mix(
    generation: GenerationMixConfig,
    entsoe: EntsoeConfig,
    raw_dir: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    """Download-or-cache-hit the generation mix (called by ingest main()).

    Reuses the cache when its manifest entry covers the requested range
    and sources; otherwise downloads year-chunked from ENTSO-E and
    persists CSV + manifest. A failed download raises (no silent
    synthetic fallback — the caller decides the fallback policy based
    on the run's overall source consistency).

    Args:
        generation: ``GenerationMixConfig`` with the configured sources.
        entsoe: ``EntsoeConfig`` with bidding zone and start date.
        raw_dir: Project raw-data directory (cache lives under
            ``<raw_dir>/entsoe/generation/``).
        start: Range start (inclusive).
        end: Range end (inclusive).

    Returns:
        A ``(df, source)`` tuple: the generation DataFrame (tz-aware
        UTC ``timestamp`` + ``{source}_mw`` columns) and the manifest
        ``source`` value actually used (``"entsoe"`` or ``"synthetic"``).

    Raises:
        ValueError: If no ENTSOE_API_KEY is set (caller falls back to
            synthetic generation for all-synthetic runs).
        RuntimeError: If the download fails for any other reason.
    """
    cache_dir = _generation_cache_dir(raw_dir)
    entry = _read_generation_manifest(cache_dir).get("generation")
    if entry is not None and _generation_entry_covers(
        entry, start, end, generation.sources
    ):
        logger.info(
            "Generation cache hit (source=%s, range=%s)",
            entry["source"],
            entry["date_range"],
        )
        df = pd.read_csv(cache_dir / _GENERATION_CSV_NAME)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df, entry["source"]

    try:
        df = download_generation_mix(
            entsoe, generation.sources, start=start, end=end
        )
        source = SOURCE_ENTSOE
    except ValueError:
        # No API key: an all-synthetic run may substitute synthetic
        # generation (the source-consistency gate in main() verifies the
        # run stays uniformly synthetic).
        logger.warning(
            "ENTSO-E generation download unavailable (no API key) — "
            "generating SYNTHETIC generation columns (NOT valid for real "
            "training runs)"
        )
        df = generate_synthetic_generation(
            pd.date_range(start=start, end=end, freq="h", tz="UTC"),
            generation.sources,
        ).reset_index(names="timestamp")
        source = SOURCE_SYNTHETIC

    _write_generation_cache(cache_dir, df, source, generation.sources)
    return df, source
