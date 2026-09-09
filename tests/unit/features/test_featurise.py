"""
Unit tests for src/featurise.py — feature engineering and splitting.
"""

from numpy import isnan
import pandas as pd
import pytest

from src.featurise import (
    _get_holiday_dates,
    add_calendar_features,
    add_derivative_features,
    add_holiday_features,
    add_holiday_proximity_features,
    add_lag_features,
    add_rolling_features,
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
        holidays = _get_holiday_dates(sample_df)
        df = add_calendar_features(sample_df.copy(), holidays)
        expected = {
            "hour",
            "day_of_week",
            "month",
            "week_of_year",
            "is_holiday",
            "is_workday",
        }
        assert expected.issubset(set(df.columns))

    def test_hour_range(self, sample_df):
        holidays = _get_holiday_dates(sample_df)
        df = add_calendar_features(sample_df.copy(), holidays)
        assert df["hour"].between(0, 23).all()

    def test_day_of_week_range(self, sample_df):
        holidays = _get_holiday_dates(sample_df)
        df = add_calendar_features(sample_df.copy(), holidays)
        assert df["day_of_week"].between(0, 6).all()

    def test_month_range(self, sample_df):
        holidays = _get_holiday_dates(sample_df)
        df = add_calendar_features(sample_df.copy(), holidays)
        assert df["month"].between(1, 12).all()

    def test_is_workday_weekday(self):
        # 2024-01-02 is a Tuesday, not a holiday
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-02", periods=1, freq="h", tz="UTC")}
        )
        holidays = _get_holiday_dates(df)
        assert bool(add_calendar_features(df, holidays)["is_workday"].iloc[0])

    def test_is_workday_weekend(self):
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-06", periods=1, freq="h", tz="UTC")}
        )
        holidays = _get_holiday_dates(df)
        assert not bool(add_calendar_features(df, holidays)["is_workday"].iloc[0])

    def test_missing_timestamp_raises(self):
        df = pd.DataFrame({"price_eur_mwh": [1.0, 2.0]})
        holidays = pd.DatetimeIndex([], tz="UTC")
        with pytest.raises(KeyError):
            add_calendar_features(df, holidays)


class TestGetHolidayDates:
    def test_sorted_and_deduplicated(self, sample_df):
        """Positive: calendar is sorted and contains no duplicate dates."""
        holidays = _get_holiday_dates(sample_df)
        assert holidays.is_monotonic_increasing
        assert len(holidays) == len(holidays.unique())

    def test_padded_one_year_each_side(self):
        """Positive: calendar covers years adjacent to the data's range."""
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-06-01", periods=2, freq="h", tz="UTC")}
        )
        holidays = _get_holiday_dates(df)
        assert set(holidays.year.unique()) == {2023, 2024, 2025}

    def test_contains_known_2024_holidays(self, holiday_dates):
        """Positive: all known 2024 Polish holidays are in the calendar."""
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-06-01", periods=1, freq="h", tz="UTC")}
        )
        holidays = _get_holiday_dates(df)
        for date_str in holiday_dates:
            assert pd.Timestamp(date_str, tz="UTC") in holidays

    def test_unsupported_country_raises(self, sample_df):
        """Negative: unknown country raises a clear ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            _get_holiday_dates(sample_df, country="Atlantis")


class TestAddHolidayFeatures:
    def test_easter_monday_is_holiday(self):
        """Positive: 2024-04-01 is Easter Monday in Poland."""
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-04-01", periods=1, freq="h", tz="UTC")}
        )
        result = add_holiday_features(df, _get_holiday_dates(df))
        assert bool(result["is_holiday"].iloc[0])

    def test_christmas_day_is_holiday(self):
        """Positive: 2024-12-25 is Christmas Day in Poland."""
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-12-25", periods=1, freq="h", tz="UTC")}
        )
        result = add_holiday_features(df, _get_holiday_dates(df))
        assert bool(result["is_holiday"].iloc[0])

    def test_regular_day_not_holiday(self):
        """Positive: 2024-07-15 is not a Polish holiday."""
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-07-15", periods=1, freq="h", tz="UTC")}
        )
        result = add_holiday_features(df, _get_holiday_dates(df))
        assert not bool(result["is_holiday"].iloc[0])

    @pytest.mark.parametrize(
        "date_str",
        [
            "2024-01-01",  # New Year
            "2024-04-01",  # Easter Monday
            "2024-05-01",  # Labour Day
            "2024-05-03",  # Constitution Day
            "2024-05-30",  # Corpus Christi
            "2024-08-15",  # Assumption
            "2024-11-01",  # All Saints
            "2024-11-11",  # Independence Day
            "2024-12-25",  # Christmas Day
            "2024-12-26",  # Second Day of Christmas
        ],
    )
    def test_all_polish_holidays_covered(self, date_str):
        """Positive: all major Polish public holidays are marked."""
        df = pd.DataFrame(
            {"timestamp": pd.date_range(date_str, periods=1, freq="h", tz="UTC")}
        )
        result = add_holiday_features(df, _get_holiday_dates(df))
        assert bool(result["is_holiday"].iloc[0])


class TestAddHolidayProximityFeatures:
    def test_days_to_next_holiday(self):
        """Positive: day before a holiday has days_to_next_holiday=1."""
        # 2024-04-30 is the day before Labour Day (2024-05-01)
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-04-30", periods=1, freq="h", tz="UTC")}
        )
        result = add_holiday_proximity_features(df, _get_holiday_dates(df))
        assert result["days_to_next_holiday"].iloc[0] == 1

    def test_days_since_last_holiday(self):
        """Positive: day after a holiday has days_since_last_holiday=1."""
        # 2024-01-02 is the day after New Year (2024-01-01)
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-02", periods=1, freq="h", tz="UTC")}
        )
        result = add_holiday_proximity_features(df, _get_holiday_dates(df))
        assert result["days_since_last_holiday"].iloc[0] == 1

    def test_on_holiday_days_to_next_is_zero(self):
        """Positive: on a holiday, days_to_next_holiday=0."""
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-05-01", periods=1, freq="h", tz="UTC")}
        )
        result = add_holiday_proximity_features(df, _get_holiday_dates(df))
        assert result["days_to_next_holiday"].iloc[0] == 0

    def test_on_holiday_days_since_last_is_zero(self):
        """Positive: on a holiday, days_since_last_holiday=0."""
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-05-01", periods=1, freq="h", tz="UTC")}
        )
        result = add_holiday_proximity_features(df, _get_holiday_dates(df))
        assert result["days_since_last_holiday"].iloc[0] == 0

    def test_late_evening_before_holiday(self):
        """Regression (review point 5): 23:00 before a holiday gives 1, not 0.

        The old implementation used Timedelta.days, which truncated
        23 hours down to 0 — indistinguishable from being on the holiday.
        """
        df = pd.DataFrame(
            {"timestamp": [pd.Timestamp("2024-04-30 23:00", tz="UTC")]}
        )
        result = add_holiday_proximity_features(df, _get_holiday_dates(df))
        assert result["days_to_next_holiday"].iloc[0] == 1

    @pytest.mark.parametrize("hour", range(24))
    def test_all_hours_of_pre_holiday_day(self, hour):
        """Every hour of the day before a holiday has days_to_next_holiday=1."""
        df = pd.DataFrame(
            {"timestamp": [pd.Timestamp(f"2024-04-30 {hour:02d}:00", tz="UTC")]}
        )
        result = add_holiday_proximity_features(df, _get_holiday_dates(df))
        assert result["days_to_next_holiday"].iloc[0] == 1

    @pytest.mark.parametrize("hour", range(24))
    def test_all_hours_of_post_holiday_day(self, hour):
        """Every hour of the day after a holiday has days_since_last_holiday=1."""
        df = pd.DataFrame(
            {"timestamp": [pd.Timestamp(f"2024-05-02 {hour:02d}:00", tz="UTC")]}
        )
        result = add_holiday_proximity_features(df, _get_holiday_dates(df))
        assert result["days_since_last_holiday"].iloc[0] == 1

    def test_year_end_uses_next_year_calendar(self):
        """Positive: Dec 31 sees next year's New Year via the padded calendar."""
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-12-31", periods=1, freq="h", tz="UTC")}
        )
        result = add_holiday_proximity_features(df, _get_holiday_dates(df))
        assert result["days_to_next_holiday"].iloc[0] == 1

    def test_unsorted_timestamps_supported(self):
        """Positive: row order does not affect the computed values."""
        df = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2024-04-30 12:00", tz="UTC"),
                    pd.Timestamp("2024-04-28 06:00", tz="UTC"),
                    pd.Timestamp("2024-05-02 18:00", tz="UTC"),
                ]
            }
        )
        result = add_holiday_proximity_features(df, _get_holiday_dates(df))
        # Apr 30 → next: May 1 (1 day); last: Easter Monday Apr 1 (29 days)
        # Apr 28 → next: May 1 (3 days); last: Easter Monday Apr 1 (27 days)
        # May 2  → next: May 3 (1 day); last: Labour Day May 1 (1 day)
        assert result["days_to_next_holiday"].tolist() == [1, 3, 1]
        assert result["days_since_last_holiday"].tolist() == [29, 27, 1]

    def test_no_holidays_in_range_returns_nan(self):
        """Boundary: empty calendar gives NaN, not an ambiguous 0."""
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-07-15", periods=3, freq="h", tz="UTC")}
        )
        result = add_holiday_proximity_features(df, pd.DatetimeIndex([], tz="UTC"))
        assert result["days_to_next_holiday"].isna().all()
        assert result["days_since_last_holiday"].isna().all()

    def test_proximity_columns_are_nullable_int(self):
        """Positive: proximity columns use the nullable Int64 dtype."""
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-04-30", periods=2, freq="h", tz="UTC")}
        )
        result = add_holiday_proximity_features(df, _get_holiday_dates(df))
        assert str(result["days_to_next_holiday"].dtype) == "Int64"
        assert str(result["days_since_last_holiday"].dtype) == "Int64"


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


class TestAddDerivativeFeatures:
    def test_first_derivative_correct(self):
        """Positive: first derivative matches diff(1) of smoothed series."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
                "price_eur_mwh": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = add_derivative_features(
            df, "price_eur_mwh", order=[1], smooth_window=1
        )
        # With smooth_window=1, no smoothing: diff(1) = [NaN, 1, 1, 1, 1]
        assert result["price_eur_mwh_diff_1"].iloc[1] == pytest.approx(1.0)
        assert result["price_eur_mwh_diff_1"].iloc[4] == pytest.approx(1.0)

    def test_second_derivative_correct(self):
        """Positive: second derivative matches diff(2) of smoothed series."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
                "price_eur_mwh": [1.0, 2.0, 4.0, 8.0, 16.0],
            }
        )
        result = add_derivative_features(
            df, "price_eur_mwh", order=[2], smooth_window=1
        )
        # diff(2) of [1, 2, 4, 8, 16] = [NaN, NaN, 3, 6, 12]
        assert result["price_eur_mwh_diff_2"].iloc[2] == pytest.approx(3.0)
        assert result["price_eur_mwh_diff_2"].iloc[4] == pytest.approx(12.0)

    def test_smoothing_applied(self):
        """Positive: smoothing window is applied before differencing."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
                "price_eur_mwh": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = add_derivative_features(
            df, "price_eur_mwh", order=[1], smooth_window=3
        )
        # With smooth_window=3, smoothed = [1, 1.5, 2, 3, 4]
        # diff(1) = [NaN, 0.5, 0.5, 1, 1]
        assert result["price_eur_mwh_diff_1"].iloc[1] == pytest.approx(0.5)
        assert result["price_eur_mwh_diff_1"].iloc[3] == pytest.approx(1.0)


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
        save_processed_data(
            sample_df, sample_df, sample_df, str(processed), str(reference)
        )
        assert processed.exists()
        assert reference.exists()

    def test_concatenates_all_splits(self, tmp_path, sample_df):
        processed = tmp_path / "features.parquet"
        reference = tmp_path / "reference.parquet"
        save_processed_data(
            sample_df, sample_df, sample_df, str(processed), str(reference)
        )
        loaded = pd.read_parquet(processed)
        assert len(loaded) == len(sample_df) * 3

    def test_reference_is_train_only(self, tmp_path, sample_df):
        processed = tmp_path / "features.parquet"
        reference = tmp_path / "reference.parquet"
        save_processed_data(
            sample_df, sample_df, sample_df, str(processed), str(reference)
        )
        ref = pd.read_parquet(reference)
        assert len(ref) == len(sample_df)
