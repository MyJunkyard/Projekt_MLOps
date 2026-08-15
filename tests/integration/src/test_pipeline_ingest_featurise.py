"""
Integration tests — raw CSV → features.parquet flow.

Exercises the interaction between src.ingest and src.featurise using
temporary files and in-memory config (no real data or params.yaml).
"""

import pandas as pd

from src.featurise import (
    add_calendar_features,
    add_lag_features,
    load_raw_data,
    save_processed_data,
    train_val_test_split,
)
from src.ingest import generate_synthetic_data, save_raw_data, validate_schema


class TestIngestToFeaturise:
    def test_full_ingest_featurise_flow(self, tmp_path, sample_config):
        """Positive: generate → validate → save → load → featurise → split → save."""
        # --- Ingest stage ---
        df = generate_synthetic_data(n_hours=200, seed=7)
        validate_schema(df)

        raw_path = tmp_path / "raw" / "synthetic.csv"
        save_raw_data(df, str(raw_path))

        # --- Featurise stage ---
        loaded = load_raw_data(str(raw_path))
        loaded = loaded.sort_values("timestamp").reset_index(drop=True)

        if sample_config["features"]["calendar"]["enabled"]:
            loaded = add_calendar_features(loaded)

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

        loaded = load_raw_data(str(raw_path)).sort_values("timestamp").reset_index(drop=True)
        loaded = add_calendar_features(loaded)
        loaded = add_lag_features(loaded, [1, 2, 24])
        loaded = loaded.dropna().reset_index(drop=True)

        # First 24 rows dropped due to lag_24h NaN
        assert len(loaded) == len(df) - 24