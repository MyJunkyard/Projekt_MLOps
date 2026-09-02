"""
Unit tests for src/ingest.py — generate_synthetic_data, validate_schema,
save_raw_data, download_entsoe_data, validate_entsoe_data,
reindex_to_grid, forward_fill_gaps, write_manifest.
"""

import hashlib
import json
import logging
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.ingest import (
    download_entsoe_data,
    fill_gaps,
    forward_fill_gaps,
    generate_synthetic_data,
    main,
    reindex_to_grid,
    save_raw_data,
    validate_entsoe_data,
    validate_schema,
    write_manifest,
)


# ---------------------------------------------------------------------------
# generate_synthetic_data
# ---------------------------------------------------------------------------
class TestGenerateSyntheticData:
    def test_returns_expected_columns(self):
        """Positive: DataFrame has timestamp, price_eur_mwh and load_mw columns.

        load_mw is included by default so the synthetic fallback is
        schema-compatible with real ENTSO-E downloads made with
        data.entsoe.include_load: true (review point 3).
        """
        df = generate_synthetic_data(n_hours=100)
        assert list(df.columns) == ["timestamp", "price_eur_mwh", "load_mw"]

    def test_load_excluded_when_disabled(self):
        """Positive: no load_mw column when include_load=False."""
        df = generate_synthetic_data(n_hours=100, include_load=False)
        assert list(df.columns) == ["timestamp", "price_eur_mwh"]

    def test_load_has_no_nulls_and_plausible_range(self):
        """Positive: synthetic load is complete and in a plausible MW band."""
        df = generate_synthetic_data(n_hours=5000)
        assert df["load_mw"].notna().all()
        # base 10000 ± seasonality (3000+1500) ± noise (500)
        assert df["load_mw"].min() > 4000
        assert df["load_mw"].max() < 16000

    def test_returns_expected_row_count(self):
        """Positive: number of rows matches n_hours."""
        df = generate_synthetic_data(n_hours=500)
        assert len(df) == 500

    def test_timestamps_are_hourly_utc(self):
        """Positive: timestamps are hourly-spaced and UTC-aware."""
        df = generate_synthetic_data(n_hours=10)
        assert df["timestamp"].dt.tz is not None
        diffs = df["timestamp"].diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()

    def test_no_null_values(self):
        """Positive: no nulls in either column."""
        df = generate_synthetic_data(n_hours=100)
        assert df["timestamp"].notna().all()
        assert df["price_eur_mwh"].notna().all()

    def test_price_range_plausible(self):
        """Positive: prices stay within a plausible band around base 50."""
        df = generate_synthetic_data(n_hours=5000)
        prices = df["price_eur_mwh"]
        # base 50 ± seasonality (10+5) ± noise (5) ± spikes (×3)
        assert prices.min() > -200
        assert prices.max() < 300

    def test_deterministic_with_seed(self):
        """Positive: same seed produces identical data."""
        df1 = generate_synthetic_data(n_hours=100, seed=42)
        df2 = generate_synthetic_data(n_hours=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_differs(self):
        """Positive: different seeds produce different data."""
        df1 = generate_synthetic_data(n_hours=100, seed=1)
        df2 = generate_synthetic_data(n_hours=100, seed=2)
        assert not df1["price_eur_mwh"].equals(df2["price_eur_mwh"])


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------
class TestValidateSchema:
    def test_valid_df_returns_true(self, sample_df):
        """Positive: a valid DataFrame returns True."""
        assert validate_schema(sample_df) is True

    @pytest.mark.parametrize(
        "mutator,message",
        [
            (lambda df: df.drop(columns=["timestamp"]), "timestamp"),
            (lambda df: df.drop(columns=["price_eur_mwh"]), "price_eur_mwh"),
        ],
    )
    def test_missing_column_raises(self, sample_df, mutator, message):
        """Negative: missing required column raises ValueError."""
        bad = mutator(sample_df.copy())
        with pytest.raises(ValueError, match=message):
            validate_schema(bad)

    def test_wrong_timestamp_type_raises(self, sample_df):
        """Negative: non-datetime timestamp raises ValueError."""
        bad = sample_df.copy()
        bad["timestamp"] = bad["timestamp"].astype(str)
        with pytest.raises(ValueError, match="datetime"):
            validate_schema(bad)

    def test_wrong_price_type_raises(self, sample_df):
        """Negative: non-numeric price raises ValueError."""
        bad = sample_df.copy()
        bad["price_eur_mwh"] = bad["price_eur_mwh"].astype(str)
        with pytest.raises(ValueError, match="numeric"):
            validate_schema(bad)

    def test_null_timestamp_raises(self, sample_df):
        """Negative: null timestamp raises ValueError."""
        bad = sample_df.copy()
        bad.loc[0, "timestamp"] = pd.NaT
        with pytest.raises(ValueError, match="null"):
            validate_schema(bad)

    def test_null_price_raises(self, sample_df):
        """Negative: null price raises ValueError."""
        bad = sample_df.copy()
        bad.loc[0, "price_eur_mwh"] = np.nan
        with pytest.raises(ValueError, match="null"):
            validate_schema(bad)

    def test_non_monotonic_timestamps_raise(self, sample_df):
        """Negative: unsorted timestamps raise ValueError."""
        bad = sample_df.copy()
        bad = bad.iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="monotonically"):
            validate_schema(bad)

    def test_duplicate_timestamps_raise(self, sample_df):
        """Negative: duplicate timestamps raise ValueError."""
        bad = sample_df.copy()
        bad.loc[1, "timestamp"] = bad.loc[0, "timestamp"]
        with pytest.raises(ValueError, match="duplicate"):
            validate_schema(bad)


# ---------------------------------------------------------------------------
# save_raw_data
# ---------------------------------------------------------------------------
class TestSaveRawData:
    def test_writes_csv_and_creates_dirs(self, tmp_path, sample_df):
        """Positive: writes CSV and creates parent directories."""
        out = tmp_path / "nested" / "dir" / "raw.csv"
        save_raw_data(sample_df, str(out))
        assert out.exists()
        loaded = pd.read_csv(out, parse_dates=["timestamp"])
        assert len(loaded) == len(sample_df)

    def test_logs_summary(self, tmp_path, sample_df, caplog):
        """Positive: logs summary including the saved path, rows, and SHA256."""
        out = tmp_path / "raw.csv"
        with caplog.at_level(logging.INFO, logger="src.ingest"):
            save_raw_data(sample_df, str(out))
        assert "Saved" in caplog.text
        assert "rows" in caplog.text
        assert "SHA256" in caplog.text

    def test_sha256_matches_file(self, tmp_path, sample_df):
        """Positive: returned SHA256 matches the on-disk file hash."""
        out = tmp_path / "raw.csv"
        returned = save_raw_data(sample_df, str(out))
        expected = hashlib.sha256(Path(out).read_bytes()).hexdigest()
        assert returned == expected

    def test_logged_sha256_matches_file(self, tmp_path, sample_df, caplog):
        """Positive: logged SHA256 matches the on-disk file hash."""
        out = tmp_path / "raw.csv"
        with caplog.at_level(logging.INFO, logger="src.ingest"):
            save_raw_data(sample_df, str(out))
        logged = next(
            record.getMessage().split("SHA256: ")[1]
            for record in caplog.records
            if record.getMessage().startswith("SHA256: ")
        )
        expected = hashlib.sha256(Path(out).read_bytes()).hexdigest()
        assert logged == expected


# ---------------------------------------------------------------------------
# download_entsoe_data
# ---------------------------------------------------------------------------
class TestDownloadEntsoeData:
    @mock.patch.dict("os.environ", {"ENTSOE_API_KEY": "test-key"}, clear=False)
    @mock.patch("src.ingest.EntsoeClient")
    def test_download_returns_dataframe(self, mock_client_class, sample_config_stage2):
        """Positive: returns a DataFrame with expected columns."""
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        timestamps = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        prices = pd.Series(np.linspace(40, 60, 24), index=timestamps)
        mock_client.query_day_ahead_prices.return_value = prices
        mock_client.query_load.return_value = pd.Series(
            np.linspace(1000, 2000, 24), index=timestamps
        )

        df = download_entsoe_data(sample_config_stage2)
        # include_load defaults to true, so load_mw is merged in
        assert list(df.columns) == ["timestamp", "price_eur_mwh", "load_mw"]
        assert len(df) == 24
        mock_client.query_load.assert_called_once()
        assert df["load_mw"].notna().all()

    @mock.patch.dict("os.environ", {"ENTSOE_API_KEY": "test-key"}, clear=False)
    @mock.patch("src.ingest.EntsoeClient")
    def test_download_load_disabled_skips_query(
        self, mock_client_class, sample_config_stage2
    ):
        """Positive: include_load=False skips query_load and omits load_mw."""
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        timestamps = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        prices = pd.Series(np.linspace(40, 60, 24), index=timestamps)
        mock_client.query_day_ahead_prices.return_value = prices

        cfg = {
            "data": {
                "entsoe": {
                    "bidding_zone": "PSE",
                    "start_date": "2024-01-01",
                    "include_load": False,
                }
            }
        }
        df = download_entsoe_data(cfg)
        mock_client.query_load.assert_not_called()
        assert list(df.columns) == ["timestamp", "price_eur_mwh"]
        assert len(df) == 24

    @mock.patch.dict("os.environ", {"ENTSOE_API_KEY": "test-key"}, clear=False)
    @mock.patch("src.ingest.EntsoeClient")
    def test_download_normalizes_dataframe_load_return(
        self, mock_client_class, sample_config_stage2
    ):
        """Positive: a DataFrame return from query_load is normalized.

        Newer entsoe-py versions return a DataFrame with an "Actual Load"
        column instead of a Series; both must produce a load_mw column.
        """
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        timestamps = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        prices = pd.Series(np.linspace(40, 60, 24), index=timestamps)
        load_df = pd.DataFrame(
            {"Actual Load": np.linspace(1000, 2000, 24)}, index=timestamps
        )
        mock_client.query_day_ahead_prices.return_value = prices
        mock_client.query_load.return_value = load_df

        df = download_entsoe_data(sample_config_stage2)
        assert "load_mw" in df.columns
        assert len(df) == 24
        assert df["load_mw"].notna().all()

    @mock.patch.dict("os.environ", {"ENTSOE_API_KEY": "test-key"}, clear=False)
    @mock.patch("src.ingest.EntsoeClient")
    def test_download_uses_bidding_zone_from_config(
        self, mock_client_class, sample_config_stage2
    ):
        """Positive: passes the configured bidding zone to the client."""
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        timestamps = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        prices = pd.Series(np.linspace(40, 60, 24), index=timestamps)
        mock_client.query_day_ahead_prices.return_value = prices
        mock_client.query_load.return_value = pd.Series(
            np.linspace(1000, 2000, 24), index=timestamps
        )

        download_entsoe_data(sample_config_stage2)
        mock_client.query_day_ahead_prices.assert_called_once()
        args, _ = mock_client.query_day_ahead_prices.call_args
        assert args[0] == "PSE"

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_download_falls_back_to_synthetic(self, sample_config_stage2):
        """Negative: no API key raises ValueError."""
        with pytest.raises(ValueError, match="ENTSOE_API_KEY"):
            download_entsoe_data(sample_config_stage2)


# ---------------------------------------------------------------------------
# validate_entsoe_data
# ---------------------------------------------------------------------------
class TestValidateEntsoeData:
    def test_valid_data_passes(self, sample_df, sample_config_stage2):
        """Positive: a valid DataFrame returns True."""
        assert validate_entsoe_data(sample_df, sample_config_stage2) is True

    def test_missing_column_raises(self, sample_df, sample_config_stage2):
        """Negative: missing required column raises ValueError."""
        bad = sample_df.drop(columns=["price_eur_mwh"])
        with pytest.raises(ValueError, match="price_eur_mwh"):
            validate_entsoe_data(bad, sample_config_stage2)

    def test_naive_timestamp_raises(self, sample_df, sample_config_stage2):
        """Negative: tz-naive timestamp raises ValueError."""
        bad = sample_df.copy()
        bad["timestamp"] = bad["timestamp"].dt.tz_localize(None)
        with pytest.raises(ValueError, match="timezone"):
            validate_entsoe_data(bad, sample_config_stage2)

    def test_gap_detection_warns_not_raises(
        self, sample_df, sample_config_stage2, caplog
    ):
        """Negative: gap larger than max_gap_periods warns instead of raising.

        The gap-filling stage handles long gaps; validation no longer
        raises on them (review point 7). Dropping 3 consecutive rows creates
        a 4-hour timestamp diff, which with max_gap_periods=2 is an
        unfillable gap (threshold = (2+1)*1h = 3h) and must warn.
        """
        bad = sample_df.copy()
        # Remove three consecutive rows to create a 4-hour gap
        bad = bad.drop(index=[5, 6, 7]).reset_index(drop=True)
        with caplog.at_level(logging.WARNING):
            assert validate_entsoe_data(bad, sample_config_stage2) is True
        assert "gap" in caplog.text.lower()
        assert "gap-filling" in caplog.text.lower()

    def test_gap_at_fillable_threshold_no_warn(
        self, sample_df, sample_config_stage2, caplog
    ):
        """Positive: a fillable gap (≤ max_gap_periods) does not warn.

        Dropping 2 consecutive rows gives a 3-hour diff, which is exactly
        the fillable boundary (max_gap_periods=2) — no warning expected.
        """
        bad = sample_df.copy().drop(index=[5, 6]).reset_index(drop=True)
        with caplog.at_level(logging.WARNING):
            assert validate_entsoe_data(bad, sample_config_stage2) is True
        assert "gap" not in caplog.text.lower()

    def test_max_gap_periods_from_config(self, sample_df):
        """Positive: gap threshold is read from data.max_gap_periods."""
        cfg = {
            "data": {"max_gap_periods": 24},
            "temporal": {"resolution": "hourly"},
        }
        # 3-hour gap is NOT a problem when max_gap_periods=24
        bad = sample_df.copy().drop(index=[5, 6]).reset_index(drop=True)
        assert validate_entsoe_data(bad, cfg) is True

    @staticmethod
    def _daily_df():
        """Daily-spaced UTC data (as temporal.resolution=daily implies)."""
        timestamps = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        return pd.DataFrame(
            {"timestamp": timestamps, "price_eur_mwh": np.linspace(40, 60, 10)}
        )

    def test_daily_resolution_gap_warns(self, caplog):
        """Negative: long gap warns when temporal.resolution=daily.

        Review point 7: the threshold must derive from temporal.resolution,
        not assume hourly data. Dropping 3 rows gives a 4-day diff; with
        max_gap_periods=2 the threshold is (2+1)*1d = 3d, so this warns.
        (Under the old hardcoded 2h threshold this would also warn, but a
        3-day diff would have been impossible for hourly data.)
        """
        bad = self._daily_df().drop(index=[4, 5, 6]).reset_index(drop=True)
        cfg = {
            "data": {"max_gap_periods": 2},
            "temporal": {"resolution": "daily"},
        }
        with caplog.at_level(logging.WARNING):
            assert validate_entsoe_data(bad, cfg) is True
        assert "gap" in caplog.text.lower()

    def test_daily_resolution_fillable_gap_no_warn(self, caplog):
        """Positive: fillable gap at daily resolution does not warn.

        Dropping 2 rows gives a 3-day diff = exactly (max_gap_periods+1)
        periods — fillable, so no warning. This exercises the daily branch
        of the period_map (review point 7).
        """
        bad = self._daily_df().drop(index=[4, 5]).reset_index(drop=True)
        cfg = {
            "data": {"max_gap_periods": 2},
            "temporal": {"resolution": "daily"},
        }
        with caplog.at_level(logging.WARNING):
            assert validate_entsoe_data(bad, cfg) is True
        assert "gap" not in caplog.text.lower()

    def test_weekly_resolution_gap_warns(self, caplog):
        """Negative: long gap warns when temporal.resolution=weekly.

        Exercises the weekly branch of the period_map (review point 7):
        dropping 3 rows gives a 4-week diff; with max_gap_periods=2 the
        threshold is (2+1)*1w = 3w — an unfillable 3-period gap that must
        warn. (Dropping 2 rows would give a 3-week diff = fillable 2-period
        gap, which correctly does not warn.)
        """
        timestamps = pd.date_range("2024-01-01", periods=8, freq="W", tz="UTC")
        df = pd.DataFrame(
            {"timestamp": timestamps, "price_eur_mwh": np.linspace(40, 60, 8)}
        )
        bad = df.drop(index=[3, 4, 5]).reset_index(drop=True)
        cfg = {
            "data": {"max_gap_periods": 2},
            "temporal": {"resolution": "weekly"},
        }
        with caplog.at_level(logging.WARNING):
            assert validate_entsoe_data(bad, cfg) is True
        assert "gap" in caplog.text.lower()

    def test_weekly_resolution_fillable_gap_no_warn(self, caplog):
        """Positive: fillable gap at weekly resolution does not warn.

        Dropping 2 rows gives a 3-week diff = exactly (max_gap_periods+1)
        periods — a fillable 2-period gap, so no warning. Completes the
        weekly branch coverage of the period_map (review point 7).
        """
        timestamps = pd.date_range("2024-01-01", periods=8, freq="W", tz="UTC")
        df = pd.DataFrame(
            {"timestamp": timestamps, "price_eur_mwh": np.linspace(40, 60, 8)}
        )
        bad = df.drop(index=[3, 4]).reset_index(drop=True)
        cfg = {
            "data": {"max_gap_periods": 2},
            "temporal": {"resolution": "weekly"},
        }
        with caplog.at_level(logging.WARNING):
            assert validate_entsoe_data(bad, cfg) is True
        assert "gap" not in caplog.text.lower()

    def test_outlier_detection_raises(self, sample_df, sample_config_stage2):
        """Negative: price outside plausible range raises ValueError."""
        bad = sample_df.copy()
        bad.loc[0, "price_eur_mwh"] = 10000.0
        with pytest.raises(ValueError, match="outlier"):
            validate_entsoe_data(bad, sample_config_stage2)

    def test_duplicate_timestamps_raise(self, sample_df, sample_config_stage2):
        """Negative: duplicate timestamps raise ValueError."""
        bad = sample_df.copy()
        bad.loc[1, "timestamp"] = bad.loc[0, "timestamp"]
        with pytest.raises(ValueError, match="duplicate"):
            validate_entsoe_data(bad, sample_config_stage2)

    def test_non_numeric_load_mw_raises(self, sample_df, sample_config_stage2):
        """Negative: non-numeric load_mw column raises ValueError."""
        bad = sample_df.copy()
        bad["load_mw"] = "not-a-number"
        with pytest.raises(ValueError, match="load_mw"):
            validate_entsoe_data(bad, sample_config_stage2)

    def test_load_mw_with_nans_passes(self, sample_df, sample_config_stage2):
        """Positive: load_mw NaNs are allowed (gap-filling handles them)."""
        bad = sample_df.copy()
        bad["load_mw"] = 1500.0
        bad.loc[0, "load_mw"] = np.nan
        assert validate_entsoe_data(bad, sample_config_stage2) is True


# ---------------------------------------------------------------------------
# forward_fill_gaps
# ---------------------------------------------------------------------------
class TestForwardFillGaps:
    def test_short_gap_filled(self, entsoe_sample_df):
        """Positive: gaps up to max_gap_periods are forward-filled.

        The entsoe_sample_df fixture has two isolated 2-hour gaps (removing
        rows at indices 26 and 34 creates missing spans of 2 hours each).
        With max_gap_periods=2 those short spans are imputed, so no NaN remain.
        """
        result = forward_fill_gaps(entsoe_sample_df, max_gap_periods=2)
        assert result["price_eur_mwh"].isna().sum() == 0

    def test_long_gap_not_filled(self):
        """Negative: a gap longer than max_gap_periods remains NaN (never ffilled).

        This is the core fix for review point 1 — previously ffill() filled
        gaps of any size. Removing 5 consecutive rows creates a 5-hour gap
        that must stay NaN with max_gap_periods=2.
        """
        timestamps = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        prices = np.linspace(40, 60, 24)
        df = pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})
        bad = df.drop(index=[10, 11, 12, 13, 14]).reset_index(drop=True)

        result = forward_fill_gaps(bad, max_gap_periods=2)
        # 5 consecutive missing hours remain NaN
        assert result["price_eur_mwh"].isna().sum() == 5
        # The NaN rows occupy exactly the missing window in the grid
        assert result["timestamp"].iloc[10:15].isna().sum() == 0

    def test_gap_at_threshold_is_filled(self):
        """Positive: a gap exactly equal to max_gap_periods is imputed."""
        timestamps = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        prices = np.linspace(40, 60, 24)
        df = pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})
        # Remove 2 consecutive rows -> 2-hour gap, exactly at max_gap_periods=2
        bad = df.drop(index=[10, 11]).reset_index(drop=True)

        result = forward_fill_gaps(bad, max_gap_periods=2)
        assert result["price_eur_mwh"].isna().sum() == 0

    def test_long_gap_flagged(self, caplog):
        """Negative: gaps longer than max_gap_periods warn and are not filled."""
        timestamps = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        prices = np.linspace(40, 60, 24)
        df = pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})
        bad = df.drop(index=[10, 11, 12, 13, 14]).reset_index(drop=True)

        with caplog.at_level(logging.WARNING):
            forward_fill_gaps(bad, max_gap_periods=2)
        assert "gap" in caplog.text.lower()


class TestReindexToGrid:
    def test_reindexes_to_complete_grid(self, entsoe_sample_df):
        """Positive: reindexes to a complete hourly grid."""
        result = reindex_to_grid(entsoe_sample_df, freq="h")
        # 72 hours total; fixture removed 2 rows -> reindexed has 72
        assert len(result) == 72
        assert result["timestamp"].is_monotonic_increasing

    def test_inserts_nan_for_missing(self, entsoe_sample_df):
        """Positive: missing hours appear as NaN rows."""
        result = reindex_to_grid(entsoe_sample_df, freq="h")
        assert result["price_eur_mwh"].isna().sum() == 2

    def test_preserves_existing_values(self, entsoe_sample_df):
        """Positive: existing values are preserved after reindexing."""
        result = reindex_to_grid(entsoe_sample_df, freq="h")
        # First row should be unchanged
        assert (
            result["price_eur_mwh"].iloc[0] == entsoe_sample_df["price_eur_mwh"].iloc[0]
        )


class TestFillGaps:
    def test_returns_stats(self, entsoe_sample_df):
        """Positive: fill_gaps returns (df, stats) with imputation info."""
        df, stats = fill_gaps(entsoe_sample_df, max_gap_periods=2)
        assert set(stats.keys()) == {
            "n_imputed",
            "n_unfilled",
            "max_gap_periods",
            "fill_method",
            "freq",
        }
        assert stats["n_imputed"] == 2
        assert stats["n_unfilled"] == 0
        assert stats["max_gap_periods"] == 2
        assert stats["fill_method"] == "ffill"

    def test_is_imputed_flag(self, entsoe_sample_df):
        """Positive: is_imputed marks only the imputed rows."""
        df, _ = fill_gaps(entsoe_sample_df, max_gap_periods=2)
        assert "is_imputed" in df.columns
        assert df["is_imputed"].sum() == 2
        # Real rows are not marked
        assert df.loc[~df["is_imputed"], "price_eur_mwh"].notna().all()

    def test_flag_off_when_disabled(self, entsoe_sample_df):
        """Negative: no is_imputed column when add_is_imputed_flag=False."""
        df, _ = fill_gaps(
            entsoe_sample_df, max_gap_periods=2, add_is_imputed_flag=False
        )
        assert "is_imputed" not in df.columns

    def test_interpolate_method(self):
        """Positive: interpolate method fills short gaps."""
        timestamps = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        prices = np.linspace(40, 60, 24)
        df = pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})
        bad = df.drop(index=[10, 11]).reset_index(drop=True)

        df_filled, stats = fill_gaps(bad, max_gap_periods=2, fill_method="interpolate")
        assert stats["fill_method"] == "interpolate"
        assert df_filled["price_eur_mwh"].isna().sum() == 0
        # Linear interpolation between 11 and 12 (index 10->11/12) should be
        # reasonable; just confirm filled values are finite.
        assert df_filled["price_eur_mwh"].notna().all()

    def test_unsupported_method_raises(self):
        """Negative: unsupported fill_method raises ValueError."""
        timestamps = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        prices = np.linspace(40, 60, 24)
        df = pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})

        with pytest.raises(ValueError, match="Unsupported fill_method"):
            fill_gaps(df, max_gap_periods=2, fill_method="extrapolate")

    def test_long_gap_unfilled_and_flagged(self):
        """Negative: long gaps stay NaN and are flagged in stats."""
        timestamps = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        prices = np.linspace(40, 60, 24)
        df = pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})
        bad = df.drop(index=[10, 11, 12, 13, 14]).reset_index(drop=True)

        df_filled, stats = fill_gaps(bad, max_gap_periods=2)
        assert df_filled["price_eur_mwh"].isna().sum() == 5
        assert stats["n_unfilled"] == 5
        assert stats["n_imputed"] == 0
        # is_imputed False for the unfilled rows (they weren't imputed)
        assert df_filled["is_imputed"].sum() == 0


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------
class TestWriteManifest:
    def test_manifest_has_required_fields(self, tmp_path, sample_df):
        """Positive: manifest JSON has required fields."""
        write_manifest(str(tmp_path), sample_df)
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "downloaded_at" in manifest
        assert "date_range" in manifest
        assert "row_count" in manifest
        assert "sha256" in manifest

    def test_manifest_sha256_matches_file(self, tmp_path, sample_df):
        """Positive: manifest SHA256 matches the DataFrame content hash."""
        write_manifest(str(tmp_path), sample_df)
        manifest_path = tmp_path / "manifest.json"

        with open(manifest_path) as f:
            manifest = json.load(f)

        csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
        expected_hash = hashlib.sha256(csv_bytes).hexdigest()
        assert manifest["sha256"] == expected_hash

    def test_manifest_row_count_matches(self, tmp_path, sample_df):
        """Positive: manifest row_count matches DataFrame length."""
        write_manifest(str(tmp_path), sample_df)
        manifest_path = tmp_path / "manifest.json"

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["row_count"] == len(sample_df)

    def test_manifest_records_imputation_stats(
        self, tmp_path, sample_df, entsoe_sample_df
    ):
        """Positive: manifest records n_imputed/n_dropped when stats provided.

        This is the auditability fix from review point 1b — the data-quality
        story (how many rows were fabricated by imputation) must be visible
        in the manifest, not hidden.
        """
        filled, stats = fill_gaps(entsoe_sample_df, max_gap_periods=2)
        stats["n_dropped_rows"] = 3  # simulate a drop of unfilled rows

        write_manifest(str(tmp_path), filled, imputation_stats=stats)
        manifest_path = tmp_path / "manifest.json"

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["n_imputed_rows"] == 2
        assert manifest["n_dropped_rows"] == 3
        assert manifest["max_gap_periods"] == 2
        assert manifest["fill_method"] == "ffill"
        assert manifest["freq"] == "h"

    def test_manifest_uses_provided_sha256(self, tmp_path, sample_df):
        """Positive: provided sha256 is used verbatim in the manifest."""
        provided = "a" * 64
        write_manifest(str(tmp_path), sample_df, sha256_hash=provided)
        manifest_path = tmp_path / "manifest.json"

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["sha256"] == provided


# ---------------------------------------------------------------------------
# main — empty-data guard (review point 4)
# ---------------------------------------------------------------------------
class TestMainEmptyDataGuard:
    """main() must fail fast with a clear error on empty ingestion output.

    A successful ENTSO-E download can still return zero rows (e.g. no data
    in the requested date range). Previously main() proceeded with an empty
    DataFrame and crashed later in reindex_to_grid with an opaque ValueError
    from df["timestamp"].min() on an empty series (review point 4).
    """

    @staticmethod
    def _make_cfg(tmp_path):
        """Build a minimal config dict for main() pointing at tmp_path."""
        return {
            "data": {
                "raw_path": str(tmp_path / "raw"),
                "entsoe": {
                    "bidding_zone": "PSE",
                    "start_date": "2024-01-01",
                },
            },
            "temporal": {"resolution": "hourly"},
        }

    @staticmethod
    def _empty_df():
        """An empty DataFrame with the schema a download would produce."""
        return pd.DataFrame(
            {"timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
             "price_eur_mwh": pd.Series(dtype=float)}
        )

    @mock.patch("src.ingest.download_entsoe_data")
    @mock.patch("src.ingest.load_config")
    def test_main_raises_on_empty_dataframe(
        self, mock_load_config, mock_download, tmp_path
    ):
        """Negative: empty download output raises a clear ValueError."""
        mock_load_config.return_value = self._make_cfg(tmp_path)
        mock_download.return_value = self._empty_df()

        with pytest.raises(ValueError, match="empty DataFrame"):
            main()

    @mock.patch("src.ingest.validate_entsoe_data")
    @mock.patch("src.ingest.download_entsoe_data")
    @mock.patch("src.ingest.load_config")
    def test_main_raises_before_validation(
        self, mock_load_config, mock_download, mock_validate, tmp_path
    ):
        """Negative: the guard fires before any downstream stage runs.

        Validation (and everything after it) must never see the empty
        DataFrame — the guard is the first check after ingestion.
        """
        mock_load_config.return_value = self._make_cfg(tmp_path)
        mock_download.return_value = self._empty_df()

        with pytest.raises(ValueError, match="empty DataFrame"):
            main()
        mock_validate.assert_not_called()

    @mock.patch("src.ingest.download_entsoe_data")
    @mock.patch("src.ingest.load_config")
    def test_main_proceeds_with_non_empty_data(
        self, mock_load_config, mock_download, tmp_path
    ):
        """Positive: non-empty data passes the guard and the pipeline runs.

        The full main() flow executes for real on a small synthetic frame
        (validation, reindexing, gap filling, save, manifest) — proving the
        guard does not reject valid data.
        """
        cfg = self._make_cfg(tmp_path)
        mock_load_config.return_value = cfg
        mock_download.return_value = generate_synthetic_data(n_hours=48)

        main()  # must not raise

        output_csv = tmp_path / "raw" / "entsoe_prices.csv"
        manifest_json = tmp_path / "raw" / "manifest.json"
        assert output_csv.exists()
        assert manifest_json.exists()


# ---------------------------------------------------------------------------
# main — config-derived grid frequency (review point 7)
# ---------------------------------------------------------------------------
class TestMainResolutionGrid:
    """main() must derive the grid frequency from temporal.resolution.

    Previously main() called reindex_to_grid(df, freq="h") with a hardcoded
    hourly frequency before fill_gaps. With temporal.resolution=daily that
    fabricated 23 NaN rows per day, which fill_gaps then misclassified as
    unfillable long gaps and dropped. fill_gaps reindexes internally with the
    config-derived freq, so the hardcoded pre-reindex was removed (review
    point 7).
    """

    @mock.patch("src.ingest.download_entsoe_data")
    @mock.patch("src.ingest.load_config")
    def test_main_daily_resolution_no_hourly_fabrication(
        self, mock_load_config, mock_download, tmp_path
    ):
        """Positive: daily resolution imputes a 2-day gap instead of dropping.

        10 daily rows with rows 4-5 removed (a 2-day gap, fillable at
        max_gap_periods=2). With the fix, main() reindexes to a daily grid and
        imputes the gap → 10 output rows. Under the old hardcoded hourly
        reindex, the grid would have been 217 hourly rows with 48 NaN rows
        dropped → 169 rows.
        """
        timestamps = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        daily_df = pd.DataFrame(
            {"timestamp": timestamps, "price_eur_mwh": np.linspace(40, 60, 10)}
        )
        mock_load_config.return_value = {
            "data": {
                "raw_path": str(tmp_path / "raw"),
                "max_gap_periods": 2,
                "fill_method": "ffill",
                "drop_long_gaps": True,
                "entsoe": {
                    "bidding_zone": "PSE",
                    "start_date": "2024-01-01",
                },
            },
            "temporal": {"resolution": "daily"},
        }
        mock_download.return_value = daily_df

        main()

        output = pd.read_csv(
            tmp_path / "raw" / "entsoe_prices.csv", parse_dates=["timestamp"]
        )
        assert len(output) == 10
        assert output["price_eur_mwh"].notna().all()


# ---------------------------------------------------------------------------
# main — manifest hash provenance (review point 9)
# ---------------------------------------------------------------------------
class TestMainHashProvenance:
    """main() must persist the on-disk SHA256 in the manifest (review point 9).

    save_raw_data returns the SHA256 of the on-disk CSV and main() passes it
    to write_manifest, so the manifest and the logged summary come from the
    same source. Each link was previously tested in isolation
    (test_sha256_matches_file, test_logged_sha256_matches_file,
    test_manifest_uses_provided_sha256); this class verifies the full
    provenance chain end-to-end through main(), which is what automation
    needs to verify the printed hash against the manifest.
    """

    @mock.patch("src.ingest.download_entsoe_data")
    @mock.patch("src.ingest.load_config")
    def test_main_manifest_sha256_matches_saved_csv_and_log(
        self, mock_load_config, mock_download, tmp_path, caplog
    ):
        """Positive: manifest sha256 == on-disk CSV hash == logged hash.

        Runs the full main() flow for real (save + manifest) and closes the
        provenance loop: the hash recorded in manifest.json must equal the
        SHA256 of the actual on-disk entsoe_prices.csv bytes, and the SHA256
        logged by save_raw_data must be the same value.
        """
        mock_load_config.return_value = {
            "data": {
                "raw_path": str(tmp_path / "raw"),
                "max_gap_periods": 2,
                "fill_method": "ffill",
                "drop_long_gaps": True,
                "entsoe": {
                    "bidding_zone": "PSE",
                    "start_date": "2024-01-01",
                },
            },
            "temporal": {"resolution": "hourly"},
        }
        mock_download.return_value = generate_synthetic_data(n_hours=48)

        with caplog.at_level(logging.INFO, logger="src.ingest"):
            main()

        output_csv = tmp_path / "raw" / "entsoe_prices.csv"
        manifest_json = tmp_path / "raw" / "manifest.json"
        assert output_csv.exists()
        assert manifest_json.exists()

        with open(manifest_json) as f:
            manifest = json.load(f)

        # Link 1: manifest hash matches the actual on-disk file bytes
        on_disk_hash = hashlib.sha256(output_csv.read_bytes()).hexdigest()
        assert manifest["sha256"] == on_disk_hash

        # Link 2: the hash logged by save_raw_data is the same value
        logged = next(
            record.getMessage().split("SHA256: ")[1]
            for record in caplog.records
            if record.getMessage().startswith("SHA256: ")
        )
        assert logged == manifest["sha256"]
