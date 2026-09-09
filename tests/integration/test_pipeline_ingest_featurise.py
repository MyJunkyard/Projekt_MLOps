"""
Integration tests — raw CSV → features.parquet flow.

Exercises the interaction between src.ingest and src.featurise using
temporary files and in-memory config (no real data or params.yaml).
"""

from unittest import mock

import pandas as pd

from src.featurise import (
    _get_holiday_dates,
    add_calendar_features,
    add_holiday_features,
    add_holiday_proximity_features,
    add_lag_features,
    add_rolling_features,
    load_raw_data,
    save_processed_data,
    train_val_test_split,
)
from src.ingest import (
    download_entsoe_data,
    generate_synthetic_data,
    save_raw_data,
    validate_schema,
)


class TestIngestToFeaturise:
    def test_full_ingest_featurise_flow(self, tmp_path, sample_config):
        """Positive: generate, validate, save, load, featurise, split, save."""
        # --- Ingest stage ---
        df = generate_synthetic_data(n_hours=200, seed=7)
        validate_schema(df)

        raw_path = tmp_path / "raw" / "synthetic.csv"
        save_raw_data(df, str(raw_path))

        # --- Featurise stage ---
        loaded = load_raw_data(str(raw_path))
        loaded = loaded.sort_values("timestamp").reset_index(drop=True)

        if sample_config["features"]["calendar"]["enabled"]:
            holidays = _get_holiday_dates(loaded)
            loaded = add_calendar_features(loaded, holidays)

        if sample_config["features"]["lags"]["enabled"]:
            periods = sample_config["features"]["lags"]["periods"]
            loaded = add_lag_features(loaded, periods)

        loaded = loaded.dropna().reset_index(drop=True)

        train, val, test = train_val_test_split(loaded, sample_config)

        processed = tmp_path / "processed" / "features.parquet"
        reference = tmp_path / "reference" / "reference.parquet"
        save_processed_data(train, val, test, str(processed), str(reference))

        # --- Assertions ---
        assert processed.exists()
        assert reference.exists()

        features = pd.read_parquet(processed)
        assert len(features) == len(train) + len(val) + len(test)
        assert "hour" in features.columns
        assert "lag_1h" in features.columns
        assert features["price_eur_mwh"].notna().all()

        ref = pd.read_parquet(reference)
        assert len(ref) == len(train)

    def test_lag_rows_dropped_before_split(self, tmp_path, sample_config):
        """Positive: NaN rows from lag creation are dropped before splitting."""
        df = generate_synthetic_data(n_hours=50, seed=1)
        raw_path = tmp_path / "raw.csv"
        save_raw_data(df, str(raw_path))

        loaded = (
            load_raw_data(str(raw_path)).sort_values("timestamp").reset_index(drop=True)
        )
        holidays = _get_holiday_dates(loaded)
        loaded = add_calendar_features(loaded, holidays)
        loaded = add_lag_features(loaded, [1, 2, 24])
        loaded = loaded.dropna().reset_index(drop=True)

        # First 24 rows dropped due to lag_24h NaN
        assert len(loaded) == len(df) - 24


class TestEntsoeToFeaturise:
    @mock.patch.dict("os.environ", {"ENTSOE_API_KEY": "test-key"}, clear=False)
    @mock.patch("src.ingest.EntsoeClient")
    def test_full_entsoe_featurise_flow(
        self, mock_client_class, tmp_path, sample_config_stage2
    ):
        """Positive: ENTSO-E download to featurise with holiday and lag columns."""
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        timestamps = pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")
        prices = pd.Series([50.0 + i % 10 for i in range(200)], index=timestamps)
        mock_client.query_day_ahead_prices.return_value = prices
        mock_client.query_load.return_value = pd.Series(
            [1000.0 + i for i in range(200)], index=timestamps
        )

        cfg = dict(sample_config_stage2)
        cfg["data"] = dict(sample_config_stage2["data"])
        cfg["data"]["train_end"] = "2020-01-05"
        cfg["data"]["val_end"] = "2020-01-08"

        df = download_entsoe_data(cfg)
        assert "timestamp" in df.columns
        assert "price_eur_mwh" in df.columns
        # include_load defaults to true: the fetched load is merged as
        # load_mw instead of being discarded (review point 3)
        assert "load_mw" in df.columns
        assert len(df) == 200

        raw_path = tmp_path / "raw" / "entsoe_prices.csv"
        save_raw_data(df, str(raw_path))

        loaded = load_raw_data(str(raw_path))
        loaded = loaded.sort_values("timestamp").reset_index(drop=True)

        if cfg["features"]["calendar"]["enabled"]:
            holidays = _get_holiday_dates(loaded)
            loaded = add_calendar_features(loaded, holidays)
            loaded = add_holiday_proximity_features(loaded, holidays)

        if cfg["features"]["lags"]["enabled"]:
            periods = cfg["features"]["lags"]["periods"]
            loaded = add_lag_features(loaded, periods)
            loaded = add_rolling_features(loaded, cfg["data"]["target_col"])

        loaded = loaded.dropna().reset_index(drop=True)

        train, val, test = train_val_test_split(loaded, cfg)

        processed = tmp_path / "processed" / "features.parquet"
        reference = tmp_path / "reference" / "reference.parquet"
        save_processed_data(train, val, test, str(processed), str(reference))

        assert processed.exists()
        features = pd.read_parquet(processed)
        assert "is_holiday" in features.columns
        assert "days_to_next_holiday" in features.columns
        assert "days_since_last_holiday" in features.columns
        assert "lag_1h" in features.columns
        assert "rolling_mean_24h" in features.columns
        # load_mw survives the full ingest → featurise flow as a feature
        assert "load_mw" in features.columns
        assert features["load_mw"].notna().all()

    def test_holiday_features_in_processed_data(self, tmp_path, sample_config_stage2):
        """Positive: holiday features are present in featurised output."""
        timestamps = pd.date_range("2024-04-28", periods=72, freq="h", tz="UTC")
        prices = [50.0 + i % 10 for i in range(72)]
        df = pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})

        raw_path = tmp_path / "raw" / "entsoe_prices.csv"
        save_raw_data(df, str(raw_path))

        loaded = load_raw_data(str(raw_path))
        holidays = _get_holiday_dates(loaded)
        loaded = add_holiday_features(loaded, holidays)
        loaded = add_holiday_proximity_features(loaded, holidays)

        assert "is_holiday" in loaded.columns
        assert "days_to_next_holiday" in loaded.columns
        assert "days_since_last_holiday" in loaded.columns
