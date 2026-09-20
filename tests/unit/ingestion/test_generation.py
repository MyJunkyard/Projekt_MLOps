"""
Unit tests for the generation-mix ingestion (Workstream 4).

Covers ``map_generation_sources`` (PSR-type → ``{source}_mw`` mapping,
MultiIndex flattening, missing-source NaN policy), the year-chunked
``download_generation_mix`` (mocked entsoe-py client), synthetic
generation fallback shapes, and the ``ingest_generation_mix``
cache discipline (hit avoids a second download, manifest provenance).
"""

import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.config.models import EntsoeConfig, GenerationMixConfig
from src.ingestion.entsoe import (
    download_generation_mix,
    generate_synthetic_data,
    generate_synthetic_generation,
    ingest_generation_mix,
    map_generation_sources,
)

_SOURCES = ["wind", "solar", "coal", "gas", "nuclear", "hydro"]


def _psr_frame(n=24):
    """Mimic an entsoe-py query_generation result (flat columns)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "Wind Offshore": np.linspace(100, 200, n),
            "Wind Onshore": np.linspace(300, 400, n),
            "Solar": np.linspace(0, 500, n),
            "Fossil Hard coal": np.linspace(4000, 5000, n),
            "Fossil Gas": np.linspace(1000, 1500, n),
            "Nuclear": np.linspace(6000, 6500, n),
            "Hydro Run-of-river and poundage": np.linspace(800, 900, n),
        },
        index=idx,
    )


def _entsoe_cfg(tmp_start="2024-01-01") -> EntsoeConfig:
    return EntsoeConfig.model_validate(
        {"bidding_zone": "PSE", "start_date": tmp_start}
    )


class TestMapGenerationSources:
    def test_maps_all_sources(self):
        mapped = map_generation_sources(_psr_frame(), _SOURCES)
        # Column order follows the requested source order
        assert list(mapped.columns) == [f"{s}_mw" for s in _SOURCES]
        assert mapped["wind_mw"].notna().all()

    def test_sums_multiple_matching_columns(self):
        """Positive: Wind Offshore + Wind Onshore sum into wind_mw."""
        frame = _psr_frame()
        expected = frame["Wind Offshore"] + frame["Wind Onshore"]
        mapped = map_generation_sources(frame, ["wind"])
        assert mapped["wind_mw"].equals(expected)

    def test_missing_source_becomes_nan_column_with_warning(self, caplog):
        frame = _psr_frame().drop(
            columns=["Nuclear", "Hydro Run-of-river and poundage"]
        )
        with caplog.at_level("WARNING"):
            mapped = map_generation_sources(frame, ["nuclear", "wind"])
        assert mapped["nuclear_mw"].isna().all()
        assert mapped["wind_mw"].notna().all()
        assert "nuclear" in caplog.text.lower()

    def test_multiindex_columns_flattened(self):
        frame = _psr_frame()
        frame.columns = pd.MultiIndex.from_tuples(
            [(c, "Actual Aggregated") for c in frame.columns]
        )
        mapped = map_generation_sources(frame, ["wind"])
        assert mapped["wind_mw"].notna().all()

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown generation source"):
            map_generation_sources(_psr_frame(), ["banana"])


class TestDownloadGenerationMix:
    @mock.patch.dict("os.environ", {"ENTSOE_API_KEY": "test-key"}, clear=False)
    @mock.patch("src.ingestion.entsoe.EntsoeClient")
    def test_year_chunking_queries_per_year(self, mock_client_class):
        """Positive: a range spanning two calendar years issues 2 chunk queries."""
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.query_generation.return_value = _psr_frame(n=24)

        df = download_generation_mix(
            _entsoe_cfg("2023-01-01"),
            _SOURCES,
            start=pd.Timestamp("2023-11-01", tz="UTC"),
            end=pd.Timestamp("2024-02-15", tz="UTC"),
        )
        # One query per calendar year: 2023 and 2024
        assert mock_client.query_generation.call_count == 2
        assert list(df.columns) == [
            "timestamp",
            *[f"{s}_mw" for s in _SOURCES],
        ]
        assert df["timestamp"].is_monotonic_increasing

    @mock.patch.dict("os.environ", {"ENTSOE_API_KEY": "test-key"}, clear=False)
    @mock.patch("src.ingestion.entsoe.EntsoeClient")
    def test_uses_bidding_zone_from_config(self, mock_client_class):
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.query_generation.return_value = _psr_frame(n=24)

        download_generation_mix(_entsoe_cfg(), _SOURCES)
        args, _ = mock_client.query_generation.call_args
        assert args[0] == "PSE"

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_no_api_key_raises(self):
        with pytest.raises(ValueError, match="ENTSOE_API_KEY"):
            download_generation_mix(_entsoe_cfg(), _SOURCES)


class TestSyntheticGeneration:
    def test_columns_match_sources(self):
        idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        gen = generate_synthetic_generation(idx, _SOURCES)
        assert list(gen.columns) == [f"{s}_mw" for s in _SOURCES]

    def test_solar_zero_at_night(self):
        """Positive: the synthetic solar shape is physically plausible."""
        idx = pd.date_range("2024-06-01", periods=24, freq="h", tz="UTC")
        gen = generate_synthetic_generation(idx, ["solar"])
        night_hours = [0, 2, 4, 22, 23]
        night_values = gen["solar_mw"].to_numpy()[night_hours]
        assert (night_values == 0.0).all()
        assert gen["solar_mw"].max() > 0

    def test_all_values_non_negative_and_finite(self):
        idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
        gen = generate_synthetic_generation(idx, _SOURCES)
        assert (gen >= 0).all().all()
        assert np.isfinite(gen.to_numpy()).all()

    def test_deterministic_with_seed(self):
        idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        gen1 = generate_synthetic_generation(idx, _SOURCES, seed=7)
        gen2 = generate_synthetic_generation(idx, _SOURCES, seed=7)
        pd.testing.assert_frame_equal(gen1, gen2)

    def test_generate_synthetic_data_schema_compatible(self):
        """Positive: fallback stays schema-compatible with generation runs."""
        df = generate_synthetic_data(
            n_hours=48, include_load=True, include_generation=True
        )
        for source in _SOURCES:
            assert f"{source}_mw" in df.columns


class TestIngestGenerationMix:
    def test_download_and_cache_write(self, tmp_path):
        """Positive: first call downloads, persists CSV + manifest, returns source."""
        with (
            mock.patch.dict("os.environ", {"ENTSOE_API_KEY": "k"}, clear=False),
            mock.patch(
                "src.ingestion.entsoe.download_generation_mix"
            ) as mock_dl,
        ):
            mock_dl.return_value = pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2024-01-01", periods=24, freq="h", tz="UTC"
                    ),
                    "wind_mw": np.linspace(100, 200, 24),
                }
            )
            df, source = ingest_generation_mix(
                GenerationMixConfig(enabled=True, sources=["wind"]),
                _entsoe_cfg(),
                tmp_path,
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-02", tz="UTC"),
            )
        assert source == "entsoe"
        assert "wind_mw" in df.columns
        manifest = json.loads(
            (tmp_path / "entsoe" / "generation" / "manifest.json").read_text()
        )["generation"]
        assert manifest["source"] == "entsoe"
        assert manifest["generation_sources"] == ["wind"]
        assert manifest["non_null_rows"]["wind_mw"] == 24

    def test_cache_hit_avoids_second_download(self, tmp_path):
        """Positive: a covering manifest entry means zero additional calls."""
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=48, freq="h", tz="UTC"
                ),
                "wind_mw": np.linspace(100, 200, 48),
            }
        )
        with (
            mock.patch.dict("os.environ", {"ENTSOE_API_KEY": "k"}, clear=False),
            mock.patch(
                "src.ingestion.entsoe.download_generation_mix",
                return_value=frame,
            ) as mock_dl,
        ):
            ingest_generation_mix(
                GenerationMixConfig(enabled=True, sources=["wind"]),
                _entsoe_cfg(),
                tmp_path,
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-02", tz="UTC"),
            )
            assert mock_dl.call_count == 1

            # Second call: same range → cache hit, no new download
            df2, source2 = ingest_generation_mix(
                GenerationMixConfig(enabled=True, sources=["wind"]),
                _entsoe_cfg(),
                tmp_path,
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-02", tz="UTC"),
            )
            assert mock_dl.call_count == 1
            assert source2 == "entsoe"
            assert len(df2) == 48

    def test_no_api_key_falls_back_to_synthetic(self, tmp_path):
        """Positive: no API key → synthetic frame cached with source=synthetic."""
        with mock.patch.dict("os.environ", {}, clear=True):
            df, source = ingest_generation_mix(
                GenerationMixConfig(enabled=True, sources=["wind", "solar"]),
                _entsoe_cfg("2024-01-01"),
                tmp_path,
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-03", tz="UTC"),
            )
        assert source == "synthetic"
        # Column order follows GenerationMixConfig.sources (sorted by validator)
        assert list(df.columns) == ["timestamp", "solar_mw", "wind_mw"]
        manifest = json.loads(
            (tmp_path / "entsoe" / "generation" / "manifest.json").read_text()
        )["generation"]
        assert manifest["source"] == "synthetic"
        # Manifest stores the sorted sources
        assert manifest["generation_sources"] == ["solar", "wind"]
