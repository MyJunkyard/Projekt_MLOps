"""
Functional tests — full pipeline smoke test.

Runs the complete pipeline (ingest → featurise → train → evaluate) on a
small synthetic dataset using temporary files and mocked MLflow.
"""

from unittest import mock

import numpy as np
import pandas as pd

from src.evaluate import load_test_features
from src.featurise import (
    add_calendar_features,
    add_lag_features,
    load_raw_data,
    save_processed_data,
    train_val_test_split,
)
from src.ingest import generate_synthetic_data, save_raw_data, validate_schema
from src.train import load_features, load_model
from src.utils import compute_metrics


class TestEndToEndPipeline:
    @mock.patch("src.train.mlflow")
    def test_full_pipeline_smoke(self, mock_mlflow, tmp_path, sample_config):
        """Positive: full pipeline runs end-to-end and produces valid outputs."""
        # generate_synthetic_data starts at 2020-01-01; align split dates to it
        cfg = dict(sample_config)
        cfg["data"] = dict(sample_config["data"])
        cfg["data"]["train_end"] = "2020-01-05"
        cfg["data"]["val_end"] = "2020-01-10"

        # --- 1. Ingest ---
        df = generate_synthetic_data(n_hours=300, seed=42)
        validate_schema(df)
        raw_path = tmp_path / "raw" / "synthetic.csv"
        save_raw_data(df, str(raw_path))

        # --- 2. Featurise ---
        loaded = load_raw_data(str(raw_path))
        loaded = loaded.sort_values("timestamp").reset_index(drop=True)
        loaded = add_calendar_features(loaded)
        loaded = add_lag_features(loaded, [1, 2, 24])
        loaded = loaded.dropna().reset_index(drop=True)

        train, val, test = train_val_test_split(loaded, cfg)
        processed = tmp_path / "processed" / "features.parquet"
        reference = tmp_path / "reference" / "reference.parquet"
        save_processed_data(train, val, test, str(processed), str(reference))

        # --- 3. Train ---
        X_train, y_train, X_val, y_val, X_test, y_test = load_features(
            str(processed), cfg
        )
        model = load_model(cfg)
        model.fit(X_train, y_train)

        # --- 4. Evaluate ---
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, cfg["evaluation"]["metrics"])

        # --- Assertions ---
        assert processed.exists()
        assert reference.exists()
        assert y_pred.shape == y_test.shape
        assert np.isfinite(y_pred).all()
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0

        # Model is a fitted DummyRegressor predicting the mean
        assert np.allclose(y_pred, y_train.mean(), atol=1e-6)