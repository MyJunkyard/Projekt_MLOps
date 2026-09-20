"""
ingestion/weather/synthetic.py — Synthetic weather substitute generator.

Deterministic, offline weather data for **tests and explicitly opted-in
offline runs only** (Workstream 3, decision D7). This module is kept
strictly separate from the real acquisition path
(``weather.openmeteo``): it shares only the location registry and the
column contract.

Frames generated here must be written to the cache with
``source="synthetic"`` (see ``weather.cache``) so they can never
silently mix with real data — a run that mixes real and synthetic
sources is rejected outright.
"""

import logging
from collections.abc import Sequence
from typing import Callable

import numpy as np
import pandas as pd

from src.ingestion.weather.locations import validate_location

# Stable module name (not `__name__` — under `python -m` it is `"__main__"`
# and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.ingestion.weather.synthetic"
logger = logging.getLogger(MODULE_LOGGER_NAME)

#: Default seed for the synthetic generator (deterministic offline runs).
DEFAULT_SEED = 42


def _synth_temperature(
    steps: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Smooth temperature: annual + diurnal cycle, plausible Celsius range."""
    annual = 8.0 * np.sin(2 * np.pi * steps / (24 * 365))
    diurnal = 5.0 * np.sin(2 * np.pi * steps / 24)
    return np.clip(
        12.0 + annual + diurnal + rng.normal(0, 1.5, len(steps)), -20.0, 38.0
    )


def _synth_wind_speed(
    steps: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Wind speed in m/s: mild seasonality, non-negative."""
    annual = 3.0 * np.sin(2 * np.pi * steps / (24 * 365) + 2.0)
    diurnal = 1.0 * np.sin(2 * np.pi * steps / 24 + 1.0)
    return np.clip(
        np.abs(6.0 + annual + diurnal + rng.normal(0, 2.0, len(steps))),
        0.0,
        25.0,
    )


def _synth_shortwave_radiation(
    steps: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shortwave radiation in W/m2: zero at night, summer-peaked."""
    hour_of_day = steps % 24
    daylight = np.clip(np.sin(np.pi * (hour_of_day - 6) / 12), 0.0, None)
    annual = 0.5 + 0.5 * np.sin(2 * np.pi * steps / (24 * 365))
    return np.clip(
        800.0 * daylight * annual + rng.normal(0, 20.0, len(steps)),
        0.0,
        1200.0,
    )


def _synth_cloud_cover(
    steps: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Cloud cover in %: bounded to [0, 100]."""
    annual = 25.0 * np.sin(2 * np.pi * steps / (24 * 365) + 1.0)
    diurnal = 15.0 * np.sin(2 * np.pi * steps / 24 + 0.5)
    return np.clip(
        50.0 + annual + diurnal + rng.normal(0, 10.0, len(steps)),
        0.0,
        100.0,
    )


def _synth_precipitation(
    steps: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Precipitation in mm: mostly dry, occasional moderate rain events."""
    rain = np.where(
        rng.random(len(steps)) < 0.05,
        rng.gamma(2.0, 1.0, len(steps)),
        0.0,
    )
    return np.clip(rain, 0.0, 50.0)


#: Per-variable synthetic generators (the full ``WeatherVariable`` set).
SYNTHETIC_GENERATORS: dict[
    str, Callable[[np.ndarray, np.random.Generator], np.ndarray]
] = {
    "temperature_2m": _synth_temperature,
    "wind_speed_100m": _synth_wind_speed,
    "shortwave_radiation": _synth_shortwave_radiation,
    "cloud_cover": _synth_cloud_cover,
    "precipitation": _synth_precipitation,
}


def generate_synthetic_weather(
    start: pd.Timestamp,
    end: pd.Timestamp,
    locations: Sequence[str],
    variables: Sequence[str],
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """Generate deterministic synthetic weather frames per location.

    Smooth sine-based values per variable with diurnal and annual cycles,
    bounded to plausible physical ranges — same discipline as the ENTSO-E
    synthetic generator in ``ingestion.entsoe``.

    Args:
        start: Range start (floored to the hour).
        end: Range end (ceiled to the hour).
        locations: Location keys (validated against ``LOCATION_COORDS``).
        variables: Variable names (must have a synthetic generator).
        seed: Base random seed; each location is offset by its index so
            different locations get different (but reproducible) series.

    Returns:
        A dict mapping each location to a DataFrame with a tz-aware UTC
        ``timestamp`` column and one float column per variable.

    Raises:
        ValueError: If a location or variable has no synthetic generator,
            or the requested range is empty.
    """
    start_ts = pd.Timestamp(start).floor("h")
    end_ts = pd.Timestamp(end).ceil("h")
    if end_ts <= start_ts:
        raise ValueError(
            f"Synthetic weather range is empty: {start_ts} .. {end_ts}"
        )
    timestamps = pd.date_range(start_ts, end_ts, freq="h", tz="UTC")
    steps = np.arange(len(timestamps), dtype=float)

    frames: dict[str, pd.DataFrame] = {}
    for index, location in enumerate(locations):
        validate_location(location)
        rng = np.random.default_rng(seed + index)
        data: dict[str, np.ndarray | pd.DatetimeIndex] = {
            "timestamp": timestamps
        }
        for variable in variables:
            generator = SYNTHETIC_GENERATORS.get(variable)
            if generator is None:
                raise ValueError(
                    f"No synthetic generator for weather variable "
                    f"{variable!r}; supported: {sorted(SYNTHETIC_GENERATORS)}"
                )
            data[variable] = generator(steps, rng)
        frames[location] = pd.DataFrame(data)
        logger.info(
            "Generated %d synthetic weather records for %s",
            len(frames[location]),
            location,
        )
    return frames
