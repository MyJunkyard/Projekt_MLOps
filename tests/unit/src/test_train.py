"""
Unit tests for src/train.py — model loading, feature loading, git hash, MLflow logging.
"""

import subprocess
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.train import get_git_commit_hash, load_features, load_model, log_to_mlflow


class TestLoadModel:
    def test_loads_dummy_regressor(self, sample_config):
        """Positive: instantiates model from fully-qualified name."""
        model = load_model(sample_config)
        assert model.__class__.__name__ == "DummyRegressor"

    def test_invalid_module_raises(self, sample_config):
        """Negative: unknown module raises ImportError."""
        cfg = dict(sample_config)
        cfg["model"] = {"type": "no.such.module.Model", "params": {}}
        with pytest.raises(ImportError):
            load_model(cfg)

    def test_invalid_class_raises(self, sample_config):
        """Negative: unknown class raises AttributeError."""
        cfg = dict(sample_config)
        cfg["model"] = {"type": "sklearn.dummy.NoSuchClass", "params": {}}
        with pytest.raises(AttributeError):
            load_model(cfg)


class TestLoadFeatures:
    def test_returns_correct_shapes(self, features_parquet_path, sample_config):
        """Positive: returns X/y splits with expected shapes."""
        X_train, y_train, X_val, y_val, X_test, y_test = load_features(
            features_parquet_path, sample_config
        )
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        # All rows accounted for
        total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        assert total == len(pd.read_parquet(features_parquet_path))

    def test_excludes_target_and_timestamp(self, features_parquet_path, sample_config):
        """Positive: feature matrix excludes target and timestamp columns."""
        X_train, *_ = load_features(features_parquet_path, sample_config)
        df = pd.read_parquet(features_parquet_path)
        feature_cols = [c for c in df.columns if c not in ["price_eur_mwh", "timestamp"]]
        assert X_train.shape[1] == len(feature_cols)

    def test_missing_file_raises(self, tmp_path, sample_config):
        """Negative: missing parquet raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_features(str(tmp_path / "nope.parquet"), sample_config)


class TestGetGitCommitHash:
    def test_returns_hash_string(self):
        """Positive: returns a non-empty string."""
        result = get_git_commit_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    @mock.patch("src.train.subprocess.run", side_effect=FileNotFoundError)
    def test_returns_unknown_when_not_git(self, mock_run):
        """Negative: returns 'unknown' when git is unavailable."""
        assert get_git_commit_hash() == "unknown"

    @mock.patch("src.train.subprocess.run", side_effect=subprocess.CalledProcessError(1, "git"))
    def test_returns_unknown_on_error(self, mock_run):
        """Negative: returns 'unknown' when git command fails."""
        assert get_git_commit_hash() == "unknown"


class TestLogToMlflow:
    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_returns_run_id(self, mock_mlflow, mock_log_model, sample_config):
        """Positive: returns run_id from started run."""
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.MlflowClient.return_value = mock.MagicMock()
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)

        model = mock.MagicMock()
        run_id = log_to_mlflow(model, {"rmse": 1.0}, sample_config)
        assert run_id == "test-run-id"

    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_logs_params_and_metrics(self, mock_mlflow, mock_log_model, sample_config):
        """Positive: logs model params, metrics, and tags."""
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "run-1"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.MlflowClient.return_value = mock.MagicMock()
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)

        model = mock.MagicMock()
        log_to_mlflow(model, {"rmse": 1.0, "mae": 0.5}, sample_config)

        mock_mlflow.log_param.assert_any_call("strategy", "mean")
        mock_mlflow.log_metric.assert_any_call("rmse", 1.0)
        mock_mlflow.log_metric.assert_any_call("mae", 0.5)
        mock_mlflow.set_tag.assert_any_call("stage", "1")
