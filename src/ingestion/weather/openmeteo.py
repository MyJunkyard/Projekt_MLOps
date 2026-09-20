"""
ingestion/weather/openmeteo.py — Open-Meteo historical weather client.

The *real data* acquisition path (Workstream 3, decision D2): hourly
historical weather from the Open-Meteo archive API with transient-failure
retries. There is deliberately no fallback here — the synthetic-substitute
policy lives in ``weather.synthetic`` and is enforced by
``weather.cache.ingest_weather`` (decision D7).
"""

import logging
import time
from collections.abc import Sequence

import pandas as pd
import requests

from src.ingestion.weather.locations import validate_location

# Stable module name (not `__name__` — under `python -m` it is `"__main__"`
# and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.ingestion.weather.openmeteo"
logger = logging.getLogger(MODULE_LOGGER_NAME)

#: Open-Meteo historical weather API endpoint (hourly archive).
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

#: ``source`` value written to the manifest for successful API downloads.
SOURCE_OPEN_METEO = "open-meteo"

#: HTTP request behaviour.
REQUEST_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0


def download_weather(
    location: str,
    variables: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Download hourly historical weather for one location from Open-Meteo.

    Retries transient failures (``MAX_ATTEMPTS`` with exponential
    backoff) before giving up.

    Args:
        location: Location key from ``weather.locations.LOCATION_COORDS``.
        variables: Open-Meteo hourly variable names.
        start: Range start (inclusive; truncated to a date by the API).
        end: Range end (inclusive; truncated to a date by the API).

    Returns:
        A DataFrame with a tz-aware UTC ``timestamp`` column and one
        float column per requested variable, sorted by timestamp.

    Raises:
        RuntimeError: If all attempts fail, or the API response does not
            contain the requested variables, or it contains duplicate
            timestamps.
    """
    latitude, longitude = validate_location(location)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": pd.Timestamp(start).date().isoformat(),
        "end_date": pd.Timestamp(end).date().isoformat(),
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                ARCHIVE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as exc:
            last_error = exc
            logger.warning(
                "Open-Meteo request attempt %d/%d failed for %s: %s",
                attempt,
                MAX_ATTEMPTS,
                location,
                exc,
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
    else:
        raise RuntimeError(
            f"Open-Meteo download failed for {location!r} after "
            f"{MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    return _parse_archive_response(payload, location, variables)


def _parse_archive_response(
    payload: dict, location: str, variables: Sequence[str]
) -> pd.DataFrame:
    """Parse an Open-Meteo archive JSON payload into the weather schema.

    Args:
        payload: Decoded JSON response body.
        location: Location name (for error messages only).
        variables: Requested hourly variable names.

    Returns:
        A DataFrame with a tz-aware UTC ``timestamp`` column and one
        float column per requested variable, sorted by timestamp.

    Raises:
        RuntimeError: If the response is missing requested data or
            contains duplicate timestamps.
    """
    hourly = payload.get("hourly") or {}
    missing = [v for v in variables if v not in hourly]
    if "time" not in hourly or missing:
        raise RuntimeError(
            f"Open-Meteo response for {location!r} is missing requested "
            f"data: {'time' if 'time' not in hourly else missing}"
        )

    df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"], utc=True)})
    for variable in variables:
        # `coerce`: the API may return nulls for e.g. radiation at night.
        df[variable] = pd.to_numeric(
            pd.Series(hourly[variable]), errors="coerce"
        ).astype(float)

    if df["timestamp"].duplicated().any():
        raise RuntimeError(
            f"Open-Meteo response for {location!r} contains duplicate "
            "timestamps — refusing to return ambiguous data."
        )
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        "Downloaded %d weather records for %s (%s)",
        len(df),
        location,
        ", ".join(variables),
    )
    return df
