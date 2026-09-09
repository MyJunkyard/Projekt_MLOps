"""
Unit tests for src/train.py — model loading, feature loading, git hash, MLflow logging,
baselines, params hash, feature importances.
"""

import copy
import subprocess
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.train import (
    PersistenceModel,
    SeasonalNaiveModel,
    compute_params_hash,
    get_feature_names,
    get_git_commit_hash,
    load_features,
    load_model,
    log_feature_importances,
    log_to_mlflow,
    train_baseline_persistence,
    train_baseline_seasonal_naive,
)


class TestLoadModel:
    def test_loads_dummy_regressor(self, sample_config):
        """Positive: instantiates model from fully-qualified name."""
        model = load_model(sample_config)
        assert model.__class__.__name__ == "DummyRegressor"

    def test_loads_xgboost_regressor(self, sample_config_stage2):
        """Positive: loads xgboost.XGBRegressor from config."""
        model = load_model(sample_config_stage2)
        assert model.__class__.__name__ == "XGBRegressor"

    def test_xgboost_params_passed(self, sample_config_stage2):
        """Positive: hyperparameters are passed to the constructor."""
        model = load_model(sample_config_stage2)
        assert model.n_estimators == 10
        assert model.max_depth == 3
        assert model.learning_rate == 0.1

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


class TestGetFeatureNames:
    def test_returns_feature_names(self, features_parquet_path, sample_config):
        """Positive: returns all feature column names."""
        names = get_feature_names(features_parquet_path, sample_config)
        assert "hour" in names
        assert "lag_1h" in names
        assert "price_eur_mwh" not in names
        assert "timestamp" not in names


class TestLoadFeatures:
    def test_returns_correct_shapes(self, features_parquet_path, sample_config):
        """Positive: returns X/y splits with expected shapes."""
        X_train, y_train, X_val, y_val, X_test, y_test = load_features(
            features_parquet_path, sample_config
        )
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        assert total == len(pd.read_parquet(features_parquet_path))

    def test_excludes_target_and_timestamp(self, features_parquet_path, sample_config):
        """Positive: feature matrix excludes target and timestamp columns."""
        X_train, *_ = load_features(features_parquet_path, sample_config)
        df = pd.read_parquet(features_parquet_path)
        feature_cols = [
            c for c in df.columns if c not in ["price_eur_mwh", "timestamp"]
        ]
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

    @mock.patch(
        "src.train.subprocess.run", side_effect=subprocess.CalledProcessError(1, "git")
    )
    def test_returns_unknown_on_error(self, mock_run):
        """Negative: returns 'unknown' when git command fails."""
        assert get_git_commit_hash() == "unknown"


class TestComputeParamsHash:
    def test_returns_sha256_string(self, tmp_path):
        """Positive: returns a 64-char hex digest."""
        cfg_path = tmp_path / "params.yaml"
        cfg_path.write_text("data:\n  target_col: price_eur_mwh\n")
        result = compute_params_hash(str(cfg_path))
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_missing_file_returns_unknown(self, tmp_path):
        """Negative: missing file returns 'unknown'."""
        assert compute_params_hash(str(tmp_path / "nope.yaml")) == "unknown"


class TestLogFeatureImportances:
    @mock.patch("src.train.mlflow")
    def test_logs_json_artifact(self, mock_mlflow):
        """Positive: logs feature importances as JSON artifact."""
        model = mock.MagicMock()
        model.feature_importances_ = np.array([0.7, 0.2, 0.1])
        log_feature_importances(model, ["f1", "f2", "f3"])

        mock_mlflow.log_artifact.assert_called_once()
        args = mock_mlflow.log_artifact.call_args[0]
        kwargs = mock_mlflow.log_artifact.call_args[1]
        # The temp file is deleted after logging; verify the call signature
        assert args[0].endswith(".json")
        assert kwargs == {"artifact_path": "feature_importances"}

    @mock.patch("src.train.mlflow")
    def test_handles_no_importances(self, mock_mlflow):
        """Negative: model without feature importances is handled gracefully."""

        class NoImportancesModel:
            def __init__(self):
                pass

        model = NoImportancesModel()
        log_feature_importances(model, ["f1", "f2"])
        mock_mlflow.log_artifact.assert_not_called()

    @mock.patch("src.train.mlflow")
    def test_empty_importances_skipped(self, mock_mlflow):
        """Negative: empty importances array is not logged."""
        model = mock.MagicMock()
        model.feature_importances_ = np.array([])
        log_feature_importances(model, ["f1", "f2"])
        assert mock_mlflow.log_artifact.call_count == 0


class TestPersistenceModel:
    def test_predict_last_value(self):
        """Positive: persistence predicts the last observed value."""
        model = PersistenceModel()
        X = np.array([[1], [2], [3]])
        model.fit(X, np.array([10.0, 20.0, 30.0]))
        preds = model.predict(np.array([[0], [0], [0]]))
        np.testing.assert_allclose(preds, np.array([30.0, 30.0, 30.0]))

    def test_feature_importances_empty(self):
        """Positive: returns empty array for feature importances."""
        model = PersistenceModel()
        assert len(model.feature_importances_) == 0

    def test_predict_accepts_dataframe(self):
        """Contract: DataFrame input gives the same predictions as ndarray."""
        model = PersistenceModel()
        X = np.array([[1.0], [2.0], [3.0]])
        model.fit(X, np.array([10.0, 20.0, 30.0]))
        df = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0]})
        preds = model.predict(df)
        np.testing.assert_allclose(preds, np.full(4, 30.0))
        assert isinstance(preds, np.ndarray)

    def test_predict_accepts_list_input(self):
        """Contract: plain nested-list input is normalized via np.asarray."""
        model = PersistenceModel()
        model.fit([[1.0], [2.0]], [10.0, 20.0])
        preds = model.predict([[0.0], [0.0], [0.0]])
        np.testing.assert_allclose(preds, np.full(3, 20.0))


class TestSeasonalNaiveModel:
    def test_predicts_seasonal_pattern(self):
        """Positive: predicts by cycling through the last N values."""
        model = SeasonalNaiveModel(season_length=3)
        X = np.array([[1], [2], [3], [4], [5], [6]])
        model.fit(X, np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
        preds = model.predict(np.array([[0], [0], [0], [0], [0]]))
        np.testing.assert_allclose(preds, np.array([40.0, 50.0, 60.0, 40.0, 50.0]))

    def test_empty_history_returns_zeros(self):
        """Negative: empty history returns zeros."""
        model = SeasonalNaiveModel(season_length=3)
        preds = model.predict(np.array([[0], [0]]))
        np.testing.assert_allclose(preds, np.zeros(2))

    def test_predict_accepts_dataframe(self):
        """Contract: DataFrame input gives the same cycling as ndarray."""
        model = SeasonalNaiveModel(season_length=3)
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        model.fit(X, np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
        df = pd.DataFrame({"a": [0.0] * 5})
        preds = model.predict(df)
        np.testing.assert_allclose(preds, np.array([40.0, 50.0, 60.0, 40.0, 50.0]))
        assert isinstance(preds, np.ndarray)


class TestTrainBaselinePersistence:
    def test_returns_model_and_metrics(self):
        """Positive: returns a PersistenceModel and metric dict."""
        X_train = np.random.rand(50, 1)
        y_train = np.random.rand(50)
        X_val = np.random.rand(10, 1)
        y_val = np.random.rand(10)

        model, metrics = train_baseline_persistence(X_train, y_train, X_val, y_val)
        assert isinstance(model, PersistenceModel)
        assert "rmse" in metrics
        assert "mae" in metrics


class TestTrainBaselineSeasonalNaive:
    def test_returns_model_and_metrics(self):
        """Positive: returns a SeasonalNaiveModel and metric dict."""
        X_train = np.random.rand(200, 1)
        y_train = np.random.rand(200)
        X_val = np.random.rand(10, 1)
        y_val = np.random.rand(10)

        model, metrics = train_baseline_seasonal_naive(X_train, y_train, X_val, y_val)
        assert isinstance(model, SeasonalNaiveModel)
        assert "rmse" in metrics
        assert "mae" in metrics


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
        mock_mlflow.set_tag.assert_any_call("stage", "2")

    @mock.patch("src.train.compute_params_hash", return_value="abc123")
    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_logs_params_hash(
        self, mock_mlflow, mock_log_model, mock_params_hash, sample_config
    ):
        """Positive: params_hash tag is logged."""
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "run-2"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.MlflowClient.return_value = mock.MagicMock()
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)
        mock_mlflow.set_tag.assert_any_call("params_hash", "abc123")

    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_logs_params_yaml_artifact(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Positive: params.yaml is logged as an artifact."""
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "run-3"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.MlflowClient.return_value = mock.MagicMock()
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)
        mock_mlflow.log_artifact.assert_any_call(mock.ANY, artifact_path="config")


class TestLogToMlflowPromotion:
    """Promotion via champion alias: only the primary model is promoted."""

    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_promotes_when_flag_set(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Positive: primary model version gets the champion alias."""
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "run-promote"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_log_model.return_value = mock.MagicMock(registered_model_version="3")

        log_to_mlflow(
            mock.MagicMock(), {"rmse": 1.0}, sample_config, promote_to_production=True
        )

        mock_mlflow.MlflowClient.return_value.set_registered_model_alias.\
            assert_called_once_with(
                name=sample_config["mlflow"]["model_name"],
                alias="champion",
                version="3",
            )

    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_baseline_not_promoted(self, mock_mlflow, mock_log_model, sample_config):
        """Negative: default (no flag) never sets the alias, even when registered."""
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "run-baseline"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_log_model.return_value = mock.MagicMock(registered_model_version="4")

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        mock_mlflow.MlflowClient.return_value.set_registered_model_alias.\
            assert_not_called()

    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_config_gate_disables_promotion(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Negative: config gate off suppresses promotion even with flag set."""
        cfg = copy.deepcopy(sample_config)
        cfg["mlflow"]["promote_to_production"] = False
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "run-gated"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_log_model.return_value = mock.MagicMock(registered_model_version="5")

        log_to_mlflow(
            mock.MagicMock(), {"rmse": 1.0}, cfg, promote_to_production=True
        )

        mock_mlflow.MlflowClient.return_value.set_registered_model_alias.\
            assert_not_called()

    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_config_gate_missing_defaults_to_promote(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Positive: missing config key defaults to promotion enabled."""
        cfg = copy.deepcopy(sample_config)
        del cfg["mlflow"]["promote_to_production"]
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "run-default"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_log_model.return_value = mock.MagicMock(registered_model_version="6")

        log_to_mlflow(
            mock.MagicMock(), {"rmse": 1.0}, cfg, promote_to_production=True
        )

        mock_mlflow.MlflowClient.return_value.set_registered_model_alias.\
            assert_called_once()

    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_unregistered_model_skips_transition(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Negative: unregistered model (version None) skips alias assignment."""
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "run-unregistered"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)

        log_to_mlflow(
            mock.MagicMock(), {"rmse": 1.0}, sample_config, promote_to_production=True
        )

        mock_mlflow.MlflowClient.return_value.set_registered_model_alias.\
            assert_not_called()

    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_custom_alias_from_config(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Positive: alias name is taken from mlflow.champion_alias."""
        cfg = copy.deepcopy(sample_config)
        cfg["mlflow"]["champion_alias"] = "prod"
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "run-alias"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_log_model.return_value = mock.MagicMock(registered_model_version="7")

        log_to_mlflow(
            mock.MagicMock(), {"rmse": 1.0}, cfg, promote_to_production=True
        )

        mock_mlflow.MlflowClient.return_value.set_registered_model_alias.\
            assert_called_once_with(
                name=cfg["mlflow"]["model_name"],
                alias="prod",
                version="7",
            )

    @mock.patch("src.train.log_model")
    @mock.patch("src.train.mlflow")
    def test_promotion_logged(
        self, mock_mlflow, mock_log_model, sample_config, caplog
    ):
        """Positive: promotion emits an INFO log record."""
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "run-log"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_log_model.return_value = mock.MagicMock(registered_model_version="8")

        with caplog.at_level("INFO", logger="src.train"):
            log_to_mlflow(
                mock.MagicMock(),
                {"rmse": 1.0},
                sample_config,
                promote_to_production=True,
            )

        assert any(
            "set as 'champion' alias" in record.message for record in caplog.records
        )
