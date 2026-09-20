"""
Unit tests for the weather ingestion package (Workstream 3).

Covers the naming contract (``weather.naming``), the Open-Meteo client
with retries (``weather.openmeteo``, network fully mocked), the synthetic
generator (``weather.synthetic``), cache orchestration and the
source-consistency policy (``weather.cache``, decision D7), and the
UTC-safe merge (``weather.merge``, decisions D5/D6/D9/D10).
"""

import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import requests

from src.config.models import WeatherConfig
from src.ingestion.manifest import SOURCE_SYNTHETIC, ensure_consistent_sources
from src.ingestion.weather import (
    LOCATION_COORDS,
    ingest_weather,
    load_weather_cache,
    merge_weather,
    parse_weather_feature_name,
    select_weather_columns,
    weather_feature_name,
)
from src.ingestion.weather.openmeteo import download_weather
from src.ingestion.weather.synthetic import generate_synthetic_weather

VARIABLES = ["temperature_2m", "wind_speed_100m"]
START = pd.Timestamp("2024-01-01 00:00", tz="UTC")
END = pd.Timestamp("2024-01-03 23:00", tz="UTC")


def _weather_cfg(**overrides) -> WeatherConfig:
    """WeatherConfig for warsaw with the two test variables."""
    fields = {
        "enabled": True,
        "variables": VARIABLES,
        "locations": ["warsaw"],
        "allow_synthetic": False,
    }
    fields.update(overrides)
    return WeatherConfig.model_validate(fields)


def _weather_frame(n=72, start="2024-01-01", variables=VARIABLES):
    """Deterministic hourly weather frame (tz-aware UTC)."""
    timestamps = pd.date_range(start=start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            **{
                variable: np.linspace(1.0, 2.0, num=n)
                for variable in variables
            },
        }
    )


def _archive_payload(frame: pd.DataFrame) -> dict:
    """Build an Open-Meteo archive JSON payload from a weather frame."""
    return {
        "hourly": {
            "time": [
                ts.strftime("%Y-%m-%dT%H:%M") for ts in frame["timestamp"]
            ],
            **{
                variable: frame[variable].tolist()
                for variable in frame.columns
                if variable != "timestamp"
            },
        }
    }


class TestNaming:
    """The ``{location}__{variable}`` contract (decision D9)."""

    def test_feature_name_format(self):
        assert weather_feature_name("temperature_2m", "warsaw") == (
            "warsaw__temperature_2m"
        )

    def test_parse_round_trip(self):
        location, variable = parse_weather_feature_name(
            "warsaw__temperature_2m"
        )
        assert (location, variable) == ("warsaw", "temperature_2m")

    def test_parse_rejects_plain_column(self):
        with pytest.raises(ValueError, match="not a weather feature"):
            parse_weather_feature_name("temperature_2m")

    def test_parse_rejects_double_separator_in_variable(self):
        with pytest.raises(ValueError, match="not a weather feature"):
            parse_weather_feature_name("warsaw__temperature__2m")

    def test_select_by_location(self):
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "price_eur_mwh",
                "warsaw__temperature_2m",
                "krakow__temperature_2m",
                "warsaw__wind_speed_100m",
            ]
        )
        assert select_weather_columns(df, location="warsaw") == [
            "warsaw__temperature_2m",
            "warsaw__wind_speed_100m",
        ]

    def test_select_by_variable(self):
        df = pd.DataFrame(
            columns=[
                "warsaw__temperature_2m",
                "krakow__temperature_2m",
                "warsaw__wind_speed_100m",
            ]
        )
        assert select_weather_columns(df, variable="temperature_2m") == [
            "warsaw__temperature_2m",
            "krakow__temperature_2m",
        ]

    def test_select_ignores_non_weather_columns(self):
        df = pd.DataFrame(columns=["timestamp", "price_eur_mwh", "lag_1h"])
        assert select_weather_columns(df) == []


class TestLocations:
    def test_unknown_location_raises_listing_valid(self):
        from src.ingestion.weather.locations import validate_location

        with pytest.raises(ValueError, match="krakow"):
            validate_location("nowhere")

    def test_registry_covers_default_locations(self):
        assert set(LOCATION_COORDS) >= {
            "warsaw",
            "krakow",
            "gdansk",
            "wroclaw",
        }


class TestDownloadWeather:
    """Open-Meteo client: parse, retries, failure modes (network mocked)."""

    @mock.patch("src.ingestion.weather.openmeteo.requests.get")
    def test_parses_response_into_expected_schema(self, mock_get):
        frame = _weather_frame()
        mock_get.return_value.json.return_value = _archive_payload(frame)
        mock_get.return_value.raise_for_status.return_value = None

        df = download_weather("warsaw", VARIABLES, START, END)

        assert str(df["timestamp"].dt.tz) == "UTC"
        assert list(df.columns) == ["timestamp", *VARIABLES]
        assert all(
            pd.api.types.is_float_dtype(dtype) for dtype in df.dtypes[1:]
        )
        assert len(df) == len(frame)
        # Request went to the archive endpoint with the right location.
        assert mock_get.call_args.kwargs["params"]["latitude"] == (
            LOCATION_COORDS["warsaw"][0]
        )
        assert mock_get.call_args.kwargs["params"]["timezone"] == "UTC"

    @mock.patch("src.ingestion.weather.openmeteo.requests.get")
    def test_missing_variable_raises(self, mock_get):
        payload = _archive_payload(_weather_frame())
        del payload["hourly"]["wind_speed_100m"]
        mock_get.return_value.json.return_value = payload
        mock_get.return_value.raise_for_status.return_value = None

        with pytest.raises(RuntimeError, match="missing requested data"):
            download_weather("warsaw", VARIABLES, START, END)

    @mock.patch("src.ingestion.weather.openmeteo.time.sleep")
    @mock.patch("src.ingestion.weather.openmeteo.requests.get")
    def test_retries_then_succeeds(self, mock_get, mock_sleep):
        frame = _weather_frame()
        failure = requests.ConnectionError("transient")
        success = mock.MagicMock()
        success.json.return_value = _archive_payload(frame)
        success.raise_for_status.return_value = None
        mock_get.side_effect = [failure, failure, success]

        df = download_weather("warsaw", VARIABLES, START, END)

        assert mock_get.call_count == 3
        assert len(df) == len(frame)

    @mock.patch("src.ingestion.weather.openmeteo.time.sleep")
    @mock.patch("src.ingestion.weather.openmeteo.requests.get")
    def test_exhausted_retries_raise_runtime_error(self, mock_get, mock_sleep):
        mock_get.side_effect = requests.ConnectionError("down")
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            download_weather("warsaw", VARIABLES, START, END)

    def test_unknown_location_raises_without_http(self):
        with mock.patch(
            "src.ingestion.weather.openmeteo.requests.get"
        ) as mock_get:
            with pytest.raises(ValueError, match="Unknown weather location"):
                download_weather("nowhere", VARIABLES, START, END)
        mock_get.assert_not_called()


class TestSyntheticWeather:
    """Deterministic, plausible synthetic frames (decision D7)."""

    def test_deterministic(self):
        first = generate_synthetic_weather(START, END, ["warsaw"], VARIABLES)
        second = generate_synthetic_weather(START, END, ["warsaw"], VARIABLES)
        pd.testing.assert_frame_equal(first["warsaw"], second["warsaw"])

    def test_schema_and_plausible_ranges(self):
        variables = [
            "temperature_2m",
            "wind_speed_100m",
            "shortwave_radiation",
            "cloud_cover",
            "precipitation",
        ]
        frames = generate_synthetic_weather(
            START, END, ["warsaw", "krakow"], variables
        )
        assert set(frames) == {"warsaw", "krakow"}
        for frame in frames.values():
            assert str(frame["timestamp"].dt.tz) == "UTC"
            assert len(frame) == 72  # 3 days hourly (inclusive end)
            assert frame["temperature_2m"].between(-20, 38).all()
            assert frame["wind_speed_100m"].between(0, 25).all()
            assert frame["shortwave_radiation"].between(0, 1200).all()
            assert frame["cloud_cover"].between(0, 100).all()
            assert frame["precipitation"].between(0, 50).all()
        # Different locations get different series (seed offset per index).
        assert not frames["warsaw"]["temperature_2m"].equals(
            frames["krakow"]["temperature_2m"]
        )

    def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="No synthetic generator"):
            generate_synthetic_weather(START, END, ["warsaw"], ["hail"])

    def test_unknown_location_raises(self):
        with pytest.raises(ValueError, match="Unknown weather location"):
            generate_synthetic_weather(START, END, ["nowhere"], VARIABLES)

    def test_empty_range_raises(self):
        with pytest.raises(ValueError, match="range is empty"):
            generate_synthetic_weather(END, START, ["warsaw"], VARIABLES)


class TestIngestWeatherCache:
    """Cache orchestration: hits, misses, and source policy (D3/D7)."""

    @mock.patch("src.ingestion.weather.cache.download_weather")
    def test_downloads_and_writes_cache_and_manifest(
        self, mock_download, tmp_path
    ):
        frame = _weather_frame()
        mock_download.return_value = frame

        frames, sources = ingest_weather(_weather_cfg(), tmp_path, START, END)

        assert mock_download.call_count == 1
        assert sources == {"warsaw": "open-meteo"}
        assert (tmp_path / "weather" / "warsaw.csv").exists()
        manifest = json.loads(
            (tmp_path / "weather" / "manifest.json").read_text()
        )
        assert manifest["warsaw"]["source"] == "open-meteo"
        assert manifest["warsaw"]["sha256"]
        assert manifest["warsaw"]["row_count"] == len(frame)
        assert manifest["warsaw"]["variables"] == VARIABLES
        assert frames["warsaw"].shape == frame.shape

    @mock.patch("src.ingestion.weather.cache.download_weather")
    def test_cache_hit_avoids_second_download(self, mock_download, tmp_path):
        mock_download.return_value = _weather_frame()

        ingest_weather(_weather_cfg(), tmp_path, START, END)
        ingest_weather(_weather_cfg(), tmp_path, START, END)

        assert mock_download.call_count == 1

    @mock.patch("src.ingestion.weather.cache.download_weather")
    def test_range_beyond_cache_triggers_redownload(
        self, mock_download, tmp_path
    ):
        mock_download.return_value = _weather_frame()
        ingest_weather(_weather_cfg(), tmp_path, START, END)

        later_end = END + pd.Timedelta(days=5)
        ingest_weather(_weather_cfg(), tmp_path, START, later_end)

        assert mock_download.call_count == 2

    @mock.patch("src.ingestion.weather.cache.download_weather")
    def test_missing_variables_trigger_redownload(
        self, mock_download, tmp_path
    ):
        mock_download.return_value = _weather_frame()
        ingest_weather(_weather_cfg(), tmp_path, START, END)

        cfg_more = _weather_cfg(variables=[*VARIABLES, "cloud_cover"])
        mock_download.return_value = _weather_frame(
            variables=[*VARIABLES, "cloud_cover"]
        )
        ingest_weather(cfg_more, tmp_path, START, END)

        assert mock_download.call_count == 2

    @mock.patch("src.ingestion.weather.cache.download_weather")
    def test_failure_without_allow_synthetic_raises(
        self, mock_download, tmp_path
    ):
        mock_download.side_effect = RuntimeError("API down")
        with pytest.raises(RuntimeError, match="allow_synthetic is false"):
            ingest_weather(_weather_cfg(), tmp_path, START, END)

    @mock.patch("src.ingestion.weather.cache.download_weather")
    def test_failure_with_allow_synthetic_caches_synthetic(
        self, mock_download, tmp_path
    ):
        mock_download.side_effect = RuntimeError("API down")
        cfg = _weather_cfg(allow_synthetic=True)

        _, sources = ingest_weather(cfg, tmp_path, START, END)

        assert sources == {"warsaw": SOURCE_SYNTHETIC}
        manifest = json.loads(
            (tmp_path / "weather" / "manifest.json").read_text()
        )
        assert manifest["warsaw"]["source"] == SOURCE_SYNTHETIC


class TestLoadWeatherCache:
    """Strict cache reader: presence, source policy, coverage (D2/D7)."""

    def _seed_cache(self, tmp_path, source="open-meteo"):
        frame = _weather_frame()
        from src.ingestion.weather.cache import _write_cached_csv

        _write_cached_csv(tmp_path / "weather", "warsaw", frame, source)
        return frame

    def test_missing_cache_raises_actionable_error(self, tmp_path):
        with pytest.raises(ValueError, match="python -m src ingest"):
            load_weather_cache(tmp_path, "warsaw", START, END, VARIABLES)

    def test_unknown_location_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown weather location"):
            load_weather_cache(tmp_path, "nowhere", START, END, VARIABLES)

    @mock.patch("src.ingestion.weather.cache.download_weather")
    def test_reads_cached_frame(self, mock_download, tmp_path):
        mock_download.return_value = _weather_frame()
        ingest_weather(_weather_cfg(), tmp_path, START, END)
        assert mock_download.call_count == 1

        loaded = load_weather_cache(tmp_path, "warsaw", START, END, VARIABLES)
        assert str(loaded["timestamp"].dt.tz) == "UTC"
        assert list(loaded.columns) == ["timestamp", *VARIABLES]
        assert mock_download.call_count == 1  # still offline

    def test_insufficient_coverage_raises(self, tmp_path):
        self._seed_cache(tmp_path)
        later_end = END + pd.Timedelta(days=30)
        with pytest.raises(ValueError, match="does not cover"):
            load_weather_cache(
                tmp_path, "warsaw", START, later_end, VARIABLES
            )

    def test_synthetic_cache_refused_without_flag(self, tmp_path):
        self._seed_cache(tmp_path, source=SOURCE_SYNTHETIC)
        with pytest.raises(RuntimeError, match="SYNTHETIC"):
            load_weather_cache(tmp_path, "warsaw", START, END, VARIABLES)

    def test_synthetic_cache_allowed_with_flag(self, tmp_path):
        self._seed_cache(tmp_path, source=SOURCE_SYNTHETIC)
        loaded = load_weather_cache(
            tmp_path, "warsaw", START, END, VARIABLES, allow_synthetic=True
        )
        assert list(loaded.columns) == ["timestamp", *VARIABLES]


class TestMergeWeather:
    """UTC-safe merge: alignment, NaN policy, naming, guards (D5/D6/D9)."""

    def test_aligns_naive_price_and_aware_weather(self):
        price = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "price_eur_mwh": np.linspace(40, 60, 24),
            }
        )
        weather = _weather_frame(n=24)

        merged = merge_weather(price, weather, "warsaw")

        assert len(merged) == len(price)
        assert "warsaw__temperature_2m" in merged.columns
        assert "warsaw__wind_speed_100m" in merged.columns
        # First-hour values matched (join aligned the naive/aware keys).
        assert merged["warsaw__temperature_2m"].iloc[0] == pytest.approx(
            weather["temperature_2m"].iloc[0]
        )

    def test_missing_hours_become_nan(self):
        price = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "price_eur_mwh": np.linspace(40, 60, 24),
            }
        )
        weather = _weather_frame(n=12)  # covers only the first 12 hours

        merged = merge_weather(price, weather, "warsaw")

        assert len(merged) == len(price)
        assert merged["warsaw__temperature_2m"].iloc[:12].notna().all()
        assert merged["warsaw__temperature_2m"].iloc[12:].isna().all()

    def test_duplicate_weather_timestamps_raise(self):
        price = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="h"),
                "price_eur_mwh": [40.0, 41.0],
            }
        )
        weather = _weather_frame(n=3)
        weather = pd.concat([weather, weather.iloc[[0]]], ignore_index=True)

        with pytest.raises(ValueError, match="duplicate timestamps"):
            merge_weather(price, weather, "warsaw")

    def test_empty_weather_raises(self):
        price = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="h"),
                "price_eur_mwh": [40.0, 41.0],
            }
        )
        with pytest.raises(ValueError, match="empty"):
            merge_weather(price, _weather_frame().iloc[0:0], "warsaw")

    def test_price_frame_not_mutated(self):
        price = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="h"),
                "price_eur_mwh": [40.0, 41.0],
            }
        )
        original_columns = list(price.columns)
        merge_weather(price, _weather_frame(n=2), "warsaw")
        assert list(price.columns) == original_columns


class TestEnsureConsistentSources:
    """The source-consistency gate (decision D7)."""

    def test_all_synthetic_passes(self):
        ensure_consistent_sources({"entsoe": "synthetic", "w": "synthetic"})

    def test_all_real_passes(self):
        ensure_consistent_sources({"entsoe": "entsoe", "w": "open-meteo"})

    def test_single_dataset_passes(self):
        ensure_consistent_sources({"entsoe": "synthetic"})

    def test_mixed_sources_raise(self):
        with pytest.raises(RuntimeError, match="mixes real and synthetic"):
            ensure_consistent_sources(
                {"entsoe": "entsoe", "weather/warsaw": "synthetic"}
            )
