"""
ingestion/weather/ — Open-Meteo historical weather acquisition.

Submodules (role-based, mirroring the project's package structure):

- ``locations`` — the location → coordinates registry (shared contract).
- ``naming`` — the ``{location}__{variable}`` column naming contract.
- ``openmeteo`` — the *real data* client (archive API, retries).
- ``synthetic`` — the synthetic substitute generator (tests / explicit
  offline opt-in only; kept strictly separate from real acquisition).
- ``cache`` — per-location CSV + manifest persistence and the two
  pipeline entry points (:func:`ingest_weather`, :func:`load_weather_cache`).
- ``merge`` — UTC-safe merge into the price frame.

Policies (see ``docs/stage3_ws3_design_decisions.md``):

- **Timestamps** are tz-aware UTC inside this package; the featurise
  input frame may be tz-naive, so the merge normalizes both join keys to
  naive UTC instants (decision D5).
- **Missing weather** becomes NaN rows (count INFO-logged); the
  featurise stage's ``dropna()`` removes them (decision D6).
- **Synthetic data** is never generated automatically: a failed download
  raises unless ``features.weather.allow_synthetic`` is true, synthetic
  frames are marked ``source="synthetic"`` in the manifest, and
  :func:`load_weather_cache` refuses them unless that flag is set
  (decision D7).
- **Leakage note (Stage 5 TODO):** historical (actual) weather is
  legitimate for training, but a deployed forecaster needs *forecast*
  weather at inference time. At serving, weather features must come from
  a forecast API, not this historical archive.
"""

from src.ingestion.weather.cache import ingest_weather, load_weather_cache
from src.ingestion.weather.locations import (
    LOCATION_COORDS,
    validate_location,
)
from src.ingestion.weather.merge import merge_weather
from src.ingestion.weather.naming import (
    parse_weather_feature_name,
    select_weather_columns,
    weather_feature_name,
)
from src.ingestion.weather.openmeteo import (
    ARCHIVE_URL,
    SOURCE_OPEN_METEO,
    download_weather,
)
from src.ingestion.weather.synthetic import generate_synthetic_weather

__all__ = [
    "ARCHIVE_URL",
    "LOCATION_COORDS",
    "SOURCE_OPEN_METEO",
    "download_weather",
    "generate_synthetic_weather",
    "ingest_weather",
    "load_weather_cache",
    "merge_weather",
    "parse_weather_feature_name",
    "select_weather_columns",
    "validate_location",
    "weather_feature_name",
]
