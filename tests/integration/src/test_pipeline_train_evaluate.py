"""
Integration tests — features.parquet → train → evaluate flow.

Exercises the interaction between src.train and src.evaluate using
temporary files and mocked MLflow (no live MLflow server).
"""

from unittest import mock

import numpy as np
import pandas as pd

from src.evaluate import load_test_features
from src.train import load_features, load_model


class TestTrainEvaluateFlow:
    def test_train_then_evaluate(self, features_parquet_path, sample_config):
        """Positive: load features → train model → evaluate on test split."""
        # --- Train stage ---
        X_train, y_train, X_val, y_val, X_test, y_test = load_features(
            features_parquet_path, sample_config
        )
        model = load_model(sample_config)
        model.fit(X_train, y_train)

        # Model is fitted and can predict
        y_pred_val = model.predict(X_val)
        assert y_pred_val.shape == y_val.shape

        # --- Evaluate stage ---
        X_test_eval, y_test_eval = load_test_features(
            features_parquet_path, sample_config
        )
        y_pred_test = model.predict(X_test_eval)
        assert y_pred_test.shape == y_test_eval.shape
        assert np.isfinite(y_pred_test).all()

    def test_train_evaluate_consistent_splits(self, features_parquet_path, sample_config):
        """Positive: train and evaluate use the same test split."""
        _, _, _, _, X_test_train, y_test_train = load_features(
            features_parquet_path, sample_config
        )
        X_test_eval, y_test_eval = load_test_features(
            features_parquet_path, sample_config
        )
        np.testing.assert_array_equal(X_test_train, X_test_eval)
        np.testing.assert_array_equal(y_test_train, y_test_eval)

    @mock.patch("src.train.mlflow")
    def test_log_to_mlflow_integration(self, mock_mlflow, features_parquet_path, sample_config):
        """Positive: full train + log_to_mlflow with mocked MLflow."""
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "integration-run"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.MlflowClient.return_value = mock.MagicMock()

        from src.train import log_to_mlflow

        X_train, y_train, X_val, y_val, _, _ = load_features(
            features_parquet_path, sample_config
        )
        model = load_model(sample_config)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        from src.utils import compute_metrics

        metrics = compute_metrics(y_val, y_pred, sample_config["evaluation"]["metrics"])
        run_id = log_to_mlflow(model, metrics, sample_config)
        assert run_id == "integration-run"
        mock_mlflow.log_metric.assert_any_call("rmse", metrics["rmse"])