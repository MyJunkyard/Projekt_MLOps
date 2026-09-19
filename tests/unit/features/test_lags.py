"""
Unit tests for lag/rolling features and their orchestration.

Covers ``features.lags`` (pure helpers), ``features.main.build_features``
(the pre-split orchestration), and the no-leakage ordering guarantee:
lags/rollings are computed on the full frame before the train/val/test
split, so split-boundary rows carry values from preceding actuals.
"""

import logging

import numpy as np
import pandas as pd
import pytest
from numpy import isnan

from src.config.models import PipelineConfig
from src.features.lags import (
    DEFAULT_ROLLING_WINDOWS,
    add_lag_features,
    add_rolling_features,
)
from src.features.main import _apply_step, build_features, train_val_test_split


def _hourly_frame(n=300, start="2023-06-01"):
    """Deterministic hourly frame with a strictly increasing target."""
    timestamps = pd.date_range(start=start, periods=n, freq="h", tz="UTC")
    prices = np.linspace(40.0, 60.0, num=n)
    return pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})


def _lags_only_config(periods=(1, 24), windows=(24, 168)):
    """Minimal valid config with only the lags group enabled."""
    return PipelineConfig.model_validate(
        {
            "data": {
                "target_col": "price_eur_mwh",
                "train_end": "2023-06-05",
                "val_end": "2023-06-08",
            },
            "features": {
                "lags": {
                    "enabled": True,
                    "periods": list(periods),
                    "rolling_windows": list(windows),
                }
            },
            "model": {"type": "sklearn.dummy.DummyRegressor"},
        }
    )


class TestAddLagFeatures:
    def test_adds_lag_columns(self, sample_df):
        df = add_lag_features(sample_df.copy(), "price_eur_mwh", periods=[1, 24])
        assert "lag_1h" in df.columns
        assert "lag_24h" in df.columns

    def test_lag_values_shifted(self, sample_df):
        df = add_lag_features(sample_df.copy(), "price_eur_mwh", periods=[1])
        assert df["lag_1h"].iloc[1] == sample_df["price_eur_mwh"].iloc[0]
        assert df["lag_1h"].iloc[5] == sample_df["price_eur_mwh"].iloc[4]

    def test_first_rows_are_nan(self, sample_df):
        df = add_lag_features(sample_df.copy(), "price_eur_mwh", periods=[2])
        assert pd.isna(df["lag_2h"].iloc[0])
        assert pd.isna(df["lag_2h"].iloc[1])
        assert not pd.isna(df["lag_2h"].iloc[2])

    def test_original_columns_preserved(self, sample_df):
        original_cols = list(sample_df.columns)
        df = add_lag_features(sample_df.copy(), "price_eur_mwh", periods=[1])
        assert list(sample_df.columns) == original_cols
        assert "lag_1h" in df.columns

    def test_uses_given_target_col(self):
        """Positive: lags follow ``target_col``, not a hardcoded name."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
                "load_mw": [10.0, 20.0, 30.0, 40.0],
            }
        )
        result = add_lag_features(df, "load_mw", periods=[2])
        assert result["lag_2h"].iloc[2] == pytest.approx(10.0)
        assert result["lag_2h"].iloc[3] == pytest.approx(20.0)

    def test_periods_deduped_and_sorted(self, sample_df):
        """Positive: duplicate/unsorted periods yield each column once, in order."""
        df = add_lag_features(
            sample_df.copy(), "price_eur_mwh", periods=[24, 1, 24, 1]
        )
        new_cols = [c for c in df.columns if c.startswith("lag_")]
        assert new_cols == ["lag_1h", "lag_24h"]


class TestAddRollingFeatures:
    def test_rolling_mean_correct(self):
        """Positive: rolling mean matches hand-computed values."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
                "price_eur_mwh": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = add_rolling_features(df, "price_eur_mwh", windows=[3])
        # Rolling mean with min_periods=1: [1, 1.5, 2, 3, 4]
        expected = [1.0, 1.5, 2.0, 3.0, 4.0]
        assert result["rolling_mean_3h"].tolist() == pytest.approx(expected)

    def test_rolling_std_correct(self):
        """Positive: rolling std matches hand-computed values."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
                "price_eur_mwh": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = add_rolling_features(df, "price_eur_mwh", windows=[3])
        # Rolling std with min_periods=1: [NaN, 0.707, 1.0, 1.0, 1.0]
        assert (
            isnan(result["rolling_std_3h"].iloc[0])
        )  # NaN
        assert result["rolling_std_3h"].iloc[2] == pytest.approx(1.0)

    def test_no_future_leakage(self):
        """Positive: rolling window does not include future rows."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
                "price_eur_mwh": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = add_rolling_features(df, "price_eur_mwh", windows=[3])
        # Row 2 (index 2) should only use rows 0-2, not row 3
        assert result["rolling_mean_3h"].iloc[2] == pytest.approx(2.0)

    def test_default_windows_match_constant(self):
        """Positive: ``windows=None`` falls back to DEFAULT_ROLLING_WINDOWS."""
        assert list(DEFAULT_ROLLING_WINDOWS) == [24, 168]
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=200, freq="h", tz="UTC"
                ),
                "price_eur_mwh": np.linspace(1.0, 5.0, num=200),
            }
        )
        result = add_rolling_features(df, "price_eur_mwh")
        for window in DEFAULT_ROLLING_WINDOWS:
            assert f"rolling_mean_{window}h" in result.columns
            assert f"rolling_std_{window}h" in result.columns

    def test_custom_windows_only_requested_columns(self):
        """Positive: custom windows produce exactly the requested columns."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=60, freq="h", tz="UTC"
                ),
                "price_eur_mwh": np.linspace(1.0, 5.0, num=60),
            }
        )
        result = add_rolling_features(df, "price_eur_mwh", windows=[48])
        assert "rolling_mean_48h" in result.columns
        assert "rolling_std_48h" in result.columns
        assert "rolling_mean_24h" not in result.columns
        assert "rolling_mean_168h" not in result.columns


class TestBuildFeatures:
    def test_lags_disabled_yields_no_lag_columns(self, sample_df, sample_config):
        """Positive: ``lags.enabled: false`` leaves no lag/rolling columns."""
        sample_config.features.lags.enabled = False
        result = build_features(
            sample_df, sample_config.features, sample_config.data.target_col
        )
        assert not any(c.startswith("lag_") for c in result.columns)
        assert not any(c.startswith("rolling_") for c in result.columns)

    def test_lags_enabled_uses_configured_periods_and_windows(
        self, sample_df, sample_config
    ):
        """Positive: generated columns follow the config exactly."""
        sample_config.features.lags.periods = [1, 48]
        sample_config.features.lags.rolling_windows = [48]
        result = build_features(
            sample_df, sample_config.features, sample_config.data.target_col
        )
        assert "lag_1h" in result.columns
        assert "lag_48h" in result.columns
        assert "rolling_mean_48h" in result.columns
        assert "rolling_std_48h" in result.columns
        assert "lag_24h" not in result.columns
        assert "rolling_mean_24h" not in result.columns

    def test_sorts_unsorted_input(self, sample_df, sample_config):
        """Positive: row order does not affect the output frame."""
        shuffled = sample_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
        result = build_features(
            shuffled, sample_config.features, sample_config.data.target_col
        )
        assert result["timestamp"].is_monotonic_increasing

    def test_apply_step_applies_fn_and_logs_added_columns(
        self, sample_df, caplog
    ):
        """Positive: _apply_step runs fn and DEBUG-logs exactly its new columns."""
        with caplog.at_level(logging.DEBUG, logger="src.features.main"):
            result = _apply_step(
                sample_df.copy(),
                "lag features",
                lambda d: add_lag_features(d, "price_eur_mwh", [1, 24]),
            )
        assert "lag_1h" in result.columns
        assert "lag_24h" in result.columns
        debug_messages = [
            rec.message for rec in caplog.records if rec.levelno == logging.DEBUG
        ]
        assert any(
            "lag_1h" in msg and "lag_24h" in msg for msg in debug_messages
        )

    def test_empty_frame_raises_actionable_error(self, sample_config):
        """Negative: lag history longer than the data fails fast, not silently."""
        tiny = _hourly_frame(n=5)
        sample_config.features.lags.periods = [168]
        with pytest.raises(ValueError, match="empty DataFrame"):
            build_features(tiny, sample_config.features, sample_config.data.target_col)


class TestNoLeakageAcrossSplits:
    """Regression: lags/rollings must be computed pre-split (full frame).

    A per-split computation would leave NaN/wrong values at split starts;
    these tests fail if feature computation ever moves after the split.
    """

    def test_split_boundary_lags_match_preceding_actuals(self):
        """Positive: first val/test rows lag the true preceding targets."""
        cfg = _lags_only_config(periods=[1, 24], windows=[24])
        featured = build_features(
            _hourly_frame(), cfg.features, cfg.data.target_col
        )
        _, val, test = train_val_test_split(featured, cfg.data)
        by_ts = featured.set_index("timestamp")["price_eur_mwh"]

        for split in (val, test):
            first_ts = split["timestamp"].iloc[0]
            assert split["lag_1h"].iloc[0] == pytest.approx(
                by_ts[first_ts - pd.Timedelta(hours=1)]
            )
            assert split["lag_24h"].iloc[0] == pytest.approx(
                by_ts[first_ts - pd.Timedelta(hours=24)]
            )

    def test_split_boundary_rolling_spans_preceding_actuals(self):
        """Positive: first val row's rolling mean covers pre-split rows."""
        cfg = _lags_only_config(periods=[1], windows=[24])
        featured = build_features(
            _hourly_frame(), cfg.features, cfg.data.target_col
        )
        _, val, _ = train_val_test_split(featured, cfg.data)
        by_ts = featured.set_index("timestamp")["price_eur_mwh"]

        first_ts = val["timestamp"].iloc[0]
        expected = by_ts[first_ts - pd.Timedelta(hours=23) : first_ts].mean()
        assert val["rolling_mean_24h"].iloc[0] == pytest.approx(expected)

    def test_no_nan_lag_values_after_split(self):
        """Positive: every split row has complete lag values (pre-split fill)."""
        cfg = _lags_only_config(periods=[1, 2, 24], windows=[24, 168])
        featured = build_features(
            _hourly_frame(), cfg.features, cfg.data.target_col
        )
        train, val, test = train_val_test_split(featured, cfg.data)
        assert len(train) > 0 and len(val) > 0 and len(test) > 0
        lag_cols = [c for c in featured.columns if c.startswith(("lag_", "rolling_"))]
        assert lag_cols  # guard: the test actually exercises lag columns
        # build_features drops every NaN row pre-split, so no split —
        # including the val/test starts — may contain a NaN lag value.
        for split in (train, val, test):
            for col in lag_cols:
                assert split[col].notna().all(), f"NaN in {col}"
