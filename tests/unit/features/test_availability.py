"""
Unit tests for the availability-alignment mechanism (Workstream 4).

Covers ``features.main.align_availability`` (pure helper), the
availability block in ``build_features`` (config gating, schema
metadata, raw-column removal), the no-leakage contract across splits,
and the ``features.availability_lags`` config validators.
"""

import numpy as np
import pandas as pd
import pytest

from src.config.models import PipelineConfig
from src.features.main import align_availability, build_features, train_val_test_split


def _hourly_frame(n=300, start="2023-06-01", with_load=True):
    """Deterministic hourly frame with target + raw external column."""
    timestamps = pd.date_range(start=start, periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price_eur_mwh": np.linspace(40.0, 60.0, num=n),
        }
    )
    if with_load:
        df["load_mw"] = np.linspace(9000.0, 11000.0, num=n)
    return df


def _config(**features_overrides) -> PipelineConfig:
    """Minimal config with lags + availability lags enabled."""
    features = {
        "lags": {"enabled": True, "periods": [1, 24], "rolling_windows": [24]},
        "availability_lags": {"load_mw": 1},
    }
    features.update(features_overrides)
    return PipelineConfig.model_validate(
        {
            "data": {
                "target_col": "price_eur_mwh",
                "train_end": "2023-06-05",
                "val_end": "2023-06-08",
            },
            "features": features,
            "model": {"type": "sklearn.dummy.DummyRegressor"},
        }
    )


class TestAlignAvailability:
    def test_shifted_values_correct(self):
        """Positive: lagged value at row N equals raw value at row N−L."""
        df = _hourly_frame(n=10)
        result = align_availability(df, "load_mw", 1)
        assert result["load_mw_lag1h"].iloc[1] == pytest.approx(
            df["load_mw"].iloc[0]
        )
        assert result["load_mw_lag1h"].iloc[5] == pytest.approx(
            df["load_mw"].iloc[4]
        )

    def test_raw_column_dropped(self):
        """Positive (leakage guard): the raw column does not survive."""
        result = align_availability(_hourly_frame(), "load_mw", 1)
        assert "load_mw" not in result.columns
        assert "load_mw_lag1h" in result.columns

    def test_first_rows_nan(self):
        result = align_availability(_hourly_frame(), "load_mw", 2)
        assert pd.isna(result["load_mw_lag2h"].iloc[0])
        assert pd.isna(result["load_mw_lag2h"].iloc[1])
        assert result["load_mw_lag2h"].notna().iloc[2]

    def test_zero_lag_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            align_availability(_hourly_frame(), "load_mw", 0)


class TestBuildFeaturesAvailability:
    def test_lag_column_present_raw_removed(self, sample_config):
        """Positive: build_features replaces raw externals with lags."""
        sample_config.features.availability_lags = {"load_mw": 1}
        df = _hourly_frame(n=400)
        result, schema = build_features(
            df, sample_config.features, sample_config.data.target_col
        )
        assert "load_mw" not in result.columns
        assert "load_mw_lag1h" in result.columns
        # Schema metadata: grouped as availability_lag with the lag recorded
        assert schema.select(group="availability_lag") == ["load_mw_lag1h"]
        spec = schema.columns["load_mw_lag1h"]
        assert spec.availability_lag_hours == 1
        assert spec.derived_from == ["load_mw"]

    def test_custom_lag_hours_honored(self, sample_config):
        sample_config.features.availability_lags = {"load_mw": 3}
        df = _hourly_frame(n=400)
        result, schema = build_features(
            df, sample_config.features, sample_config.data.target_col
        )
        assert "load_mw_lag3h" in result.columns
        assert schema.columns["load_mw_lag3h"].availability_lag_hours == 3

    def test_configured_but_absent_column_skipped(self, sample_config):
        """Positive: load_mw absent from raw data (include_load=false case)."""
        sample_config.features.availability_lags = {"load_mw": 1, "wind_mw": 1}
        df = _hourly_frame(n=400, with_load=False)
        result, schema = build_features(
            df, sample_config.features, sample_config.data.target_col
        )
        assert "wind_mw_lag1h" not in result.columns
        assert not schema.select(group="availability_lag")

    def test_no_nan_lag_after_dropna(self, sample_config):
        """Positive: the shift's leading NaNs are removed pre-split."""
        sample_config.features.availability_lags = {"load_mw": 1}
        result, _ = build_features(
            _hourly_frame(n=400),
            sample_config.features,
            sample_config.data.target_col,
        )
        assert result["load_mw_lag1h"].notna().all()

    def test_no_leakage_across_splits(self):
        """Regression: first val/test rows carry pre-split actuals."""
        cfg = _config()
        raw = _hourly_frame(n=400)
        raw_by_ts = raw.set_index("timestamp")["load_mw"]
        featured, _ = build_features(raw, cfg.features, cfg.data.target_col)
        _, val, test = train_val_test_split(featured, cfg.data)
        # load_mw_lag1h(t) must equal the RAW load_mw one hour before t —
        # including at split boundaries where a per-split computation
        # would have produced NaN or a leaked current-hour value.
        for split in (val, test):
            first_ts = split["timestamp"].iloc[0]
            assert split["load_mw_lag1h"].iloc[0] == pytest.approx(
                raw_by_ts[first_ts - pd.Timedelta(hours=1)]
            )


class TestAvailabilityLagsConfig:
    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown column"):
            PipelineConfig.model_validate(
                {
                    "data": {
                        "target_col": "price_eur_mwh",
                        "train_end": "2023-12-31",
                        "val_end": "2024-01-01",
                    },
                    "features": {"availability_lags": {"loadmw": 1}},
                    "model": {"type": "sklearn.dummy.DummyRegressor"},
                }
            )

    def test_zero_lag_rejected(self):
        with pytest.raises(ValueError):
            PipelineConfig.model_validate(
                {
                    "data": {
                        "target_col": "price_eur_mwh",
                        "train_end": "2023-12-31",
                        "val_end": "2024-01-01",
                    },
                    "features": {"availability_lags": {"load_mw": 0}},
                    "model": {"type": "sklearn.dummy.DummyRegressor"},
                }
            )

    def test_generation_source_without_lag_rejected(self):
        """Negative: enabled generation sources must declare their lag."""
        with pytest.raises(ValueError, match="wind_mw"):
            PipelineConfig.model_validate(
                {
                    "data": {
                        "target_col": "price_eur_mwh",
                        "train_end": "2023-12-31",
                        "val_end": "2024-01-01",
                    },
                    "features": {
                        "generation_mix": {
                            "enabled": True,
                            "sources": ["wind", "solar"],
                        },
                        "availability_lags": {"load_mw": 1, "wind_mw": 1},
                    },
                    "model": {"type": "sklearn.dummy.DummyRegressor"},
                }
            )

    def test_generation_sources_covered_passes(self):
        cfg = PipelineConfig.model_validate(
            {
                "data": {
                    "target_col": "price_eur_mwh",
                    "train_end": "2023-12-31",
                    "val_end": "2024-01-01",
                },
                "features": {
                    "generation_mix": {
                        "enabled": True,
                        "sources": ["wind", "solar"],
                    },
                    "availability_lags": {
                        "load_mw": 1,
                        "wind_mw": 1,
                        "solar_mw": 1,
                    },
                },
                "model": {"type": "sklearn.dummy.DummyRegressor"},
            }
        )
        assert cfg.features.availability_lags["solar_mw"] == 1
