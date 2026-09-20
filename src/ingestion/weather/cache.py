"""
ingestion/weather/cache.py — Weather cache and ingestion orchestration.

Persistence for acquired weather data (per-location CSV + manifest with
full provenance) and the two pipeline entry points (Workstream 3,
decisions D2/D3/D7):

- :func:`ingest_weather` — called by ``ingestion/main.py``. Download-or-
  cache-hit per location; on download failure raises unless
  ``features.weather.allow_synthetic`` is explicitly true, in which case
  a WARNING-logged synthetic substitute (``source="synthetic"``) is
  cached instead.
- :func:`load_weather_cache` — called by ``features/main.py``. A pure
  cache reader: validates presence, source policy, and coverage before
  reading. The featurise stage never touches the network.
"""

import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config.models import WeatherConfig
from src.ingestion.manifest import SOURCE_SYNTHETIC, save_raw_data
from src.ingestion.weather.locations import validate_location
from src.ingestion.weather.openmeteo import (
    SOURCE_OPEN_METEO,
    download_weather,
)
from src.ingestion.weather.synthetic import generate_synthetic_weather

# Stable module name (not `__name__` — under `python -m` it is `"__main__"`
# and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.ingestion.weather.cache"
logger = logging.getLogger(MODULE_LOGGER_NAME)

_WEATHER_DIR_NAME = "weather"
_MANIFEST_NAME = "manifest.json"


def _cache_dir_for(raw_dir: str | Path) -> Path:
    """Return the weather cache directory under ``raw_dir``."""
    return Path(raw_dir) / _WEATHER_DIR_NAME


def _read_weather_manifest(cache_dir: Path) -> dict:
    """Read the per-location weather manifest (empty dict when absent)."""
    manifest_path = cache_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_weather_manifest_entry(
    cache_dir: Path, location: str, entry: dict
) -> None:
    """Add/update one location's entry in the weather manifest."""
    manifest = _read_weather_manifest(cache_dir)
    manifest[location] = entry
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / _MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Weather manifest updated: %s", manifest_path)


def _entry_covers(
    entry: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    variables: Sequence[str],
) -> bool:
    """Check whether a manifest entry covers the requested range/variables.

    Malformed entries are treated as a cache miss rather than crashing.
    """
    try:
        cached_start = pd.Timestamp(entry["date_range"][0])
        cached_end = pd.Timestamp(entry["date_range"][1])
        return bool(
            cached_start <= pd.Timestamp(start)
            and cached_end >= pd.Timestamp(end)
            and set(variables) <= set(entry["variables"])
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def _read_cached_csv(cache_dir: Path, location: str) -> pd.DataFrame:
    """Load one location's cached CSV (timestamp parsed to tz-aware UTC)."""
    csv_path = cache_dir / f"{location}.csv"
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _write_cached_csv(
    cache_dir: Path, location: str, df: pd.DataFrame, source: str
) -> dict:
    """Persist one location's weather frame; returns its manifest entry."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"{location}.csv"
    csv_hash = save_raw_data(df, str(csv_path))
    entry = {
        "source": source,
        "sha256": csv_hash,
        "date_range": [
            df["timestamp"].min().isoformat(),
            df["timestamp"].max().isoformat(),
        ],
        "row_count": len(df),
        "variables": [c for c in df.columns if c != "timestamp"],
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_weather_manifest_entry(cache_dir, location, entry)
    return entry


def ingest_weather(
    cfg: WeatherConfig,
    raw_dir: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Download or cache-hit weather for every configured location.

    Called by ``ingestion/main.py`` (decision D2). For each location in
    ``cfg.locations``: reuse the cache when its manifest entry covers the
    requested range, variables, and source policy; otherwise download
    from Open-Meteo and persist CSV + manifest entry. A failed download
    raises unless ``cfg.allow_synthetic`` is true, in which case a
    WARNING-logged synthetic frame (manifest ``source="synthetic"``) is
    generated instead (decision D7).

    Args:
        cfg: ``WeatherConfig`` with locations, variables, and the
            ``allow_synthetic`` flag.
        raw_dir: Project raw-data directory (cache lives under
            ``<raw_dir>/weather/``).
        start: Range start (inclusive).
        end: Range end (inclusive).

    Returns:
        A ``(frames, sources)`` tuple: ``frames`` maps each location to
        its weather DataFrame (tz-aware UTC ``timestamp`` + float
        variable columns); ``sources`` maps each location to the
        manifest ``source`` value actually used (``"open-meteo"`` or
        ``"synthetic"``) so the caller can run the source-consistency
        gate (decision D7).

    Raises:
        RuntimeError: If a download fails and ``allow_synthetic`` is
            false, or the download fails in a non-transient way (bad
            response payload).
    """
    cache_dir = _cache_dir_for(raw_dir)
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    for location in cfg.locations:
        entry = _read_weather_manifest(cache_dir).get(location)
        entry_is_synthetic = bool(
            entry is not None and entry.get("source") == SOURCE_SYNTHETIC
        )
        if (
            entry is not None
            and _entry_covers(entry, start, end, cfg.variables)
            and (entry_is_synthetic <= cfg.allow_synthetic)
        ):
            logger.info(
                "Weather cache hit for %s (source=%s)",
                location,
                entry["source"],
            )
            frames[location] = _read_cached_csv(cache_dir, location)
            continue

        try:
            df = download_weather(location, cfg.variables, start, end)
            source = SOURCE_OPEN_METEO
        except RuntimeError as exc:
            if not cfg.allow_synthetic:
                raise RuntimeError(
                    f"Weather download failed for {location!r} and "
                    "features.weather.allow_synthetic is false — fix the "
                    "connection/API or set the flag explicitly to generate "
                    "SYNTHETIC data (not valid for real training runs). "
                    f"Underlying error: {exc}"
                ) from exc
            logger.warning(
                "Open-Meteo download failed for %s: %s", location, exc
            )
            logger.warning(
                "SYNTHETIC weather will be generated for %s — NOT valid "
                "for real training/evaluation runs.",
                location,
            )
            df = generate_synthetic_weather(
                start, end, [location], cfg.variables
            )[location]
            source = SOURCE_SYNTHETIC

        _write_cached_csv(cache_dir, location, df, source)
        frames[location] = df
        sources[location] = source
    return frames, sources


def load_weather_cache(
    raw_dir: str | Path,
    location: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    variables: Sequence[str],
    allow_synthetic: bool = False,
) -> pd.DataFrame:
    """Read one location's weather cache (pure reader — no network).

    Called by ``features/main.py`` (decision D2). Validates the manifest
    entry before reading: cache present, source policy, coverage of the
    requested range and variables.

    Args:
        raw_dir: Project raw-data directory.
        location: Location key from ``weather.locations.LOCATION_COORDS``.
        start: Required range start (inclusive).
        end: Required range end (inclusive).
        variables: Required variable columns.
        allow_synthetic: Whether a ``source="synthetic"`` cache may be
            read (refused otherwise so synthetic data can never silently
            reach training features — decision D7).

    Returns:
        The cached weather DataFrame (tz-aware UTC ``timestamp`` + float
        variable columns).

    Raises:
        ValueError: If the location is unknown, the cache is missing, or
            it does not cover the requested range/variables.
        RuntimeError: If the cached data is synthetic and
            ``allow_synthetic`` is false.
    """
    validate_location(location)
    cache_dir = _cache_dir_for(raw_dir)
    entry = _read_weather_manifest(cache_dir).get(location)
    if entry is None:
        raise ValueError(
            f"No weather cache for {location!r} under {cache_dir} — run "
            "`python -m src ingest` with features.weather.enabled: true."
        )
    if entry.get("source") == SOURCE_SYNTHETIC and not allow_synthetic:
        raise RuntimeError(
            f"Weather cache for {location!r} is SYNTHETIC — refusing to "
            "build training/evaluation features from it. Re-run "
            "`python -m src ingest` with network access, or set "
            "features.weather.allow_synthetic: true explicitly."
        )
    if not _entry_covers(entry, start, end, variables):
        raise ValueError(
            f"Weather cache for {location!r} does not cover "
            f"{pd.Timestamp(start)} .. {pd.Timestamp(end)} with variables "
            f"{variables} (cached range: {entry.get('date_range')}, "
            f"cached variables: {entry.get('variables')}) — re-run "
            "`python -m src ingest`."
        )
    df = _read_cached_csv(cache_dir, location)
    missing_columns = [c for c in variables if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Weather cache for {location!r} is missing columns "
            f"{missing_columns} — re-run `python -m src ingest`."
        )
    logger.info("Loaded weather cache for %s (%d rows)", location, len(df))
    return df
