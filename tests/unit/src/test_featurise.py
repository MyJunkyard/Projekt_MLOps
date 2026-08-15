"""
Unit tests for src/featurise.py — feature engineering and splitting.
"""

import numpy as np
import pandas as pd
import pytest

from src.featurise import (
    add_calendar_features,
    add_lag_features,
    load_raw_data,
    save_processed_data,
    train_val_test_split,
)


class TestLoadRawData:
    def test_loads_csv_with_timestamp(self, raw_csv_path, sample_df):
        df = load_raw_data(raw_csv_path)
        assert list(df.columns) == ["timestamp", "price_eur_mwh"]
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert len(df) == len(sample_df)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw_data(str(tmp_path / "nope.csv"))


class TestAddCalendarFeatures:
    def test_adds_expected_columns(self, sample_df):
        df = add_calendar_features(sample_df.copy())
        expected = {
            "hour", "day_of_week", "month", "week_of_year",
            "is_holiday", "is_workday",
        }
        assert expected.issubset(set(df.columns))

    def test_hour_range(self, sample_df):
        df = add_calendar_features(sample_df.copy())
        assert df["hour"].between(0, 23).all()

    def test_day_of_week_range(self, sample_df):
        df = add_calendar_features(sample_df.copy())
        assert df["day_of_week"].between(0, 6).all()

    def test_month_range(self, sample_df):
        df = add_calendar_features(sample_df.copy())
        assert df["month"].between(1, 12).all()

    def test_is_holiday_false(self, sample_df):
        df = add_calendar_features(sample_df.copy())
        assert not df["is_holiday"].any()

    def test_is_workday_weekday(self):
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")}
        )
        assert bool(add_calendar_features(df)["is_workday"].iloc[0])

    def test_is_workday_weekend(self):
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-06", periods=1, freq="h", tz="UTC")}
        )
        assert not bool(add_calendar_features(df)["is_workday"].iloc[0])

    def test_missing_timestamp_raises(self):
        df = pd.DataFrame({"price_eur_mwh": [1.0, 2.0]})
        with pytest.raises(KeyError):
            add_calendar_features(df)


class TestAddLagFeatures:
    def test_adds_lag_columns(self, sample_df):
        df = add_lag_features(sample_df.copy(), periods=[1, 24])
        assert "lag_1h" in df.columns
        assert "lag_24h" in df.columns

    def test_lag_values_shifted(self, sample_df):
        df = add_lag_features(sample_df.copy(), periods=[1])
        assert df["lag_1h"].iloc[1] == sample_df["price_eur_mwh"].iloc[0]
        assert df["lag_1h"].iloc[5] == sample_df["price_eur_mwh"].iloc[4]

    def test_first_rows_are_nan(self, sample_df):
        df = add_lag_features(sample_df.copy(), periods=[2])
        assert pd.isna(df["lag_2h"].iloc[0])
        assert pd.isna(df["lag_2h"].iloc[1])
        assert not pd.isna(df["lag_2h"].iloc[2])

    def test_original_columns_preserved(self, sample_df):
        original_cols = list(sample_df.columns)
        df = add_lag_features(sample_df.copy(), periods=[1])
        assert list(sample_df.columns) == original_cols
        assert "lag_1h" in df.columns


class TestTrainValTestSplit:
    def test_splits_are_disjoint_and_cover_all(self, sample_df, sample_config):
        train, val, test = train_val_test_split(sample_df, sample_config)
        assert len(train) + len(val) + len(test) == len(sample_df)

    def test_split_preserves_order(self, sample_df, sample_config):
        train, val, test = train_val_test_split(sample_df, sample_config)
        assert train["timestamp"].is_monotonic_increasing
        assert val["timestamp"].is_monotonic_increasing
        assert test["timestamp"].is_monotonic_increasing

    def test_returns_copies(self, sample_df, sample_config):
        train, val, test = train_val_test_split(sample_df, sample_config)
        train.loc[0, "price_eur_mwh"] = -999.0
        assert sample_df.loc[0, "price_eur_mwh"] != -999.0


class TestSaveProcessedData:
    def test_writes_parquet_files(self, tmp_path, sample_df):
        processed = tmp_path / "processed" / "features.parquet"
        reference = tmp_path / "reference" / "reference.parquet"
        save_processed_data(sample_df, sample_df, sample_df, str(processed), str(reference))
        assert processed.exists()
        assert reference.exists()

    def test_concatenates_all_splits(self, tmp_path, sample_df):
        processed = tmp_path / "features.parquet"
        reference = tmp_path / "reference.parquet"
        save_processed_data(sample_df, sample_df, sample_df, str(processed), str(reference))
        loaded = pd.read_parquet(processed)
        assert len(loaded) == len(sample_df) * 3

    def test_reference_is_train_only(self, tmp_path, sample_df):
        processed = tmp_path / "features.parquet"
        reference = tmp_path / "reference.parquet"
        save_processed_data(sample_df, sample_df, sample_df, str(processed), str(reference))
        ref = pd.read_parquet(reference)
        assert len(ref) == len(sample_df)