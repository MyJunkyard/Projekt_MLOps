"""
Integration tests — features.parquet → train → evaluate flow.

Exercises the interaction between src.train and src.evaluate using
temporary files and mocked MLflow (no live MLflow server).
"""

from unittest import mock

import numpy as np

from src.evaluate import load_test_features
from src.train import (
    load_features,
    load_model,
    train_baseline_persistence,
    train_baseline_seasonal_naive,
)


class TestTrainEvaluateFlow:
    def test_train_then_evaluate(self, features_parquet_path, sample_config):
        """Positive: load features, train model, evaluate on test split."""
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

    def test_train_evaluate_consistent_splits(
        self, features_parquet_path, sample_config
    ):
        """Positive: train and evaluate use the same test split."""
        _, _, _, _, X_test_train, y_test_train = load_features(
            features_parquet_path, sample_config
        )
        X_test_eval, y_test_eval = load_test_features(
            features_parquet_path, sample_config
        )
        np.testing.assert_array_equal(X_test_train, X_test_eval)
        np.testing.assert_array_equal(y_test_train, y_test_eval)

    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_log_to_mlflow_integration(
        self, mock_mlflow, mock_log_model, features_parquet_path, sample_config
    ):
        """Positive: full train + log_to_mlflow with mocked MLflow."""
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "integration-run"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.MlflowClient.return_value = mock.MagicMock()
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)

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


class TestXGBoostVsBaselines:
    def test_xgboost_produces_finite_predictions(
        self, features_parquet_path, sample_config_stage2
    ):
        """Positive: XGBoost fit + predict succeeds with finite metrics."""
        X_train, y_train, X_val, y_val, _, _ = load_features(
            features_parquet_path, sample_config_stage2
        )

        model = load_model(sample_config_stage2)
        model.fit(X_train, y_train)
        y_pred_xgb = model.predict(X_val)

        from src.utils import compute_metrics

        xgb_metrics = compute_metrics(y_val, y_pred_xgb, ["rmse", "mae"])
        assert np.isfinite(xgb_metrics["rmse"])
        assert xgb_metrics["rmse"] >= 0.0

    def test_baselines_produce_reasonable_metrics(
        self, features_parquet_path, sample_config_stage2
    ):
        """Positive: baselines train and produce finite metrics."""
        X_train, y_train, X_val, y_val, _, _ = load_features(
            features_parquet_path, sample_config_stage2
        )

        _, persistence_metrics = train_baseline_persistence(
            X_train, y_train, X_val, y_val
        )
        _, seasonal_metrics = train_baseline_seasonal_naive(
            X_train, y_train, X_val, y_val
        )

        assert np.isfinite(persistence_metrics["rmse"])
        assert np.isfinite(seasonal_metrics["rmse"])

    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_feature_importances_logged(
        self, mock_mlflow, mock_log_model, features_parquet_path, sample_config_stage2
    ):
        """Positive: feature importances JSON artifact is logged."""
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "integration-run-2"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.MlflowClient.return_value = mock.MagicMock()
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)

        from src.train import get_feature_names, log_to_mlflow

        X_train, y_train, X_val, y_val, _, _ = load_features(
            features_parquet_path, sample_config_stage2
        )
        feature_names = get_feature_names(features_parquet_path, sample_config_stage2)

        model = load_model(sample_config_stage2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        from src.utils import compute_metrics

        metrics = compute_metrics(
            y_val, y_pred, sample_config_stage2["evaluation"]["metrics"]
        )
        log_to_mlflow(model, metrics, sample_config_stage2, feature_names=feature_names)

        # Verify feature importances were logged
        assert mock_mlflow.log_artifact.called


class TestOnlyPrimaryModelPromoted:
    @mock.patch("src.train.load_config")
    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_only_primary_model_promoted(
        self, mock_mlflow, mock_log_model, mock_load_config,
        features_parquet_path, sample_config_stage2,
    ):
        """Positive: main() sets the champion alias exactly once (xgboost only)."""
        from src.train import main as train_main

        mock_load_config.return_value = sample_config_stage2
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "integration-run-promote"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_log_model.return_value = mock.MagicMock(registered_model_version="1")

        train_main()

        # Three models are registered (xgboost + 2 baselines) but only the
        # primary model is promoted to the champion alias
        alias_calls = mock_mlflow.MlflowClient.return_value.\
            set_registered_model_alias.call_args_list
        assert len(alias_calls) == 1
        call = alias_calls[0]
        assert call.kwargs["name"] == sample_config_stage2["mlflow"]["model_name"]
        assert call.kwargs["alias"] == "champion"
        assert call.kwargs["version"] == "1"
