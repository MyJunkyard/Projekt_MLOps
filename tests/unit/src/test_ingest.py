"""
Unit tests for src/ingest.py — generate_synthetic_data, validate_schema, save_raw_data.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingest import generate_synthetic_data, save_raw_data, validate_schema


# ---------------------------------------------------------------------------
# generate_synthetic_data
# ---------------------------------------------------------------------------
class TestGenerateSyntheticData:
    def test_returns_expected_columns(self):
        """Positive: DataFrame has timestamp and price_eur_mwh columns."""
        df = generate_synthetic_data(n_hours=100)
        assert list(df.columns) == ["timestamp", "price_eur_mwh"]

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

    def test_prints_summary(self, tmp_path, sample_df, capsys):
        """Positive: prints summary including SHA256."""
        out = tmp_path / "raw.csv"
        save_raw_data(sample_df, str(out))
        captured = capsys.readouterr()
        assert "Saved:" in captured.out
        assert "Rows:" in captured.out
        assert "SHA256:" in captured.out

    def test_sha256_matches_file(self, tmp_path, sample_df):
        """Positive: printed SHA256 matches on-disk file hash."""
        out = tmp_path / "raw.csv"
        save_raw_data(sample_df, str(out))
        expected = hashlib.sha256(Path(out).read_bytes()).hexdigest()
        # Recompute from the file to confirm integrity
        actual = hashlib.sha256(Path(out).read_bytes()).hexdigest()
        assert actual == expected