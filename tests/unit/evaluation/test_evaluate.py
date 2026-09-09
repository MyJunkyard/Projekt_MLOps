"""
Unit tests for src/evaluate.py — test feature loading, results table,
plots, residual breakdown, and MLflow result logging.
"""

import logging
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    get_model_run_id,
    load_test_features,
    log_evaluation_results,
    log_results_table,
    plot_actual_vs_predicted,
    residual_breakdown,
)


class TestLoadTestFeatures:
    def test_returns_x_and_y(self, features_parquet_path, sample_config):
        """Positive: returns X_test and y_test arrays."""
        X_test, y_test = load_test_features(features_parquet_path, sample_config)
        assert X_test.shape[0] == y_test.shape[0]
        assert X_test.shape[0] > 0

    def test_missing_file_raises(self, tmp_path, sample_config):
        """Negative: missing parquet raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_test_features(str(tmp_path / "nope.parquet"), sample_config)


class TestLogResultsTable:
    def test_logs_metrics(self, caplog):
        """Positive: logs metrics at INFO level."""
        with caplog.at_level(logging.INFO, logger="src.evaluate"):
            log_results_table({"rmse": 1.2345, "mae": 0.9876})
        assert "Test rmse: 1.2345" in caplog.text
        assert "Test mae: 0.9876" in caplog.text


class TestPlotActualVsPredicted:
    def test_creates_png_file(self, tmp_path):
        """Positive: PNG file is created."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])
        out = tmp_path / "plot.png"
        result = plot_actual_vs_predicted(y_true, y_pred, str(out))
        assert out.exists()
        assert result == str(out)
        # PNG files start with the PNG magic bytes
        with open(out, "rb") as f:
            header = f.read(4)
        assert header == b"\x89PNG"

    def test_plot_has_correct_dimensions(self, tmp_path):
        """Positive: plot uses expected figure size (14x6)."""
        y_true = np.random.rand(50)
        y_pred = np.random.rand(50)
        out = tmp_path / "plot.png"
        plot_actual_vs_predicted(y_true, y_pred, str(out))
        # Verify the figure dimensions: 14x6 inches at 150 dpi = 2100x900 pixels
        from PIL import Image
        with Image.open(out) as img:
            width, height = img.size
        assert width == 2100, f"Expected width 2100, got {width}"
        assert height == 900, f"Expected height 900, got {height}"


class TestResidualBreakdown:
    def test_breakdown_by_hour(self):
        """Positive: groups by hour and computes mean absolute residual."""
        df = pd.DataFrame(
            {
                "hour": [0, 0, 1, 1, 2, 2],
                "price_eur_mwh": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        y_pred = np.array([15.0, 15.0, 35.0, 35.0, 55.0, 55.0])
        cfg = {"evaluation": {"residual_breakdown": ["hour"]}}

        result = residual_breakdown(df, y_true, y_pred, cfg)
        assert "grouping" in result.columns
        assert "category" in result.columns
        assert "mean_abs_residual" in result.columns
        assert len(result) == 3  # hours 0, 1, 2
        # Hour 0: |10-15| + |20-15| / 2 = (5+5)/2 = 5
        hour0 = result[result["category"] == 0]
        assert hour0["mean_abs_residual"].iloc[0] == pytest.approx(5.0)

    def test_breakdown_by_is_holiday(self):
        """Positive: breakdown by is_holiday column."""
        df = pd.DataFrame(
            {
                "is_holiday": [False, False, True, True],
                "price_eur_mwh": [10.0, 20.0, 30.0, 40.0],
            }
        )
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 35.0, 45.0])
        cfg = {"evaluation": {"residual_breakdown": ["is_holiday"]}}

        result = residual_breakdown(df, y_true, y_pred, cfg)
        assert len(result) == 2
        # Holidays have mean residual of (5+5)/2 = 5
        holiday_row = result[result["category"] == True]  # noqa: E712
        assert holiday_row["mean_abs_residual"].iloc[0] == pytest.approx(5.0)

    def test_returns_dataframe(self):
        """Positive: returns a DataFrame."""
        df = pd.DataFrame({"hour": [0, 1]})
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        cfg = {"evaluation": {"residual_breakdown": ["hour"]}}

        result = residual_breakdown(df, y_true, y_pred, cfg)
        assert isinstance(result, pd.DataFrame)


class TestGetModelRunId:
    @mock.patch("src.evaluate.mlflow")
    def test_returns_run_id_from_alias(self, mock_mlflow, sample_config):
        """Positive: resolves the training run ID from the champion alias."""
        mock_client = mock_mlflow.MlflowClient.return_value
        mock_client.get_model_version_by_alias.return_value.run_id = "train-run-1"

        run_id = get_model_run_id(sample_config)

        assert run_id == "train-run-1"
        mock_client.get_model_version_by_alias.assert_called_once_with(
            sample_config["mlflow"]["model_name"],
            sample_config["serving"]["model_alias"],
        )

    @mock.patch("src.evaluate.mlflow")
    def test_returns_none_on_registry_error(self, mock_mlflow, sample_config):
        """Negative: registry failure returns None instead of raising."""
        mock_mlflow.MlflowClient.return_value.get_model_version_by_alias.\
            side_effect = RuntimeError("registry down")

        assert get_model_run_id(sample_config) is None


class TestLogEvaluationResults:
    @mock.patch("src.evaluate.get_model_run_id", return_value="train-run-1")
    @mock.patch("src.evaluate.mlflow")
    def test_logs_to_training_run(self, mock_mlflow, mock_run_id, sample_config):
        """Positive: metrics and artifacts go to the training run."""
        log_evaluation_results(
            sample_config, {"rmse": 1.0}, ["plot.png", "breakdown.csv"]
        )

        mock_mlflow.start_run.assert_called_once_with(run_id="train-run-1")
        mock_mlflow.log_metric.assert_any_call("rmse", 1.0)
        logged = [call[0][0] for call in mock_mlflow.log_artifact.call_args_list]
        assert logged == ["plot.png", "breakdown.csv"]

    @mock.patch("src.evaluate.get_model_run_id", return_value=None)
    @mock.patch("src.evaluate.mlflow")
    def test_falls_back_to_named_run(self, mock_mlflow, mock_run_id, sample_config):
        """Negative: unresolved training run falls back to a named run."""
        log_evaluation_results(sample_config, {"rmse": 1.0}, [])

        mock_mlflow.start_run.assert_called_once_with(run_name="evaluation")
        mock_mlflow.log_metric.assert_any_call("rmse", 1.0)
        mock_mlflow.log_artifact.assert_not_called()

    @mock.patch("src.evaluate.get_model_run_id", return_value="train-run-1")
    @mock.patch("src.evaluate.mlflow")
    def test_empty_artifacts_ok(self, mock_mlflow, mock_run_id, sample_config):
        """Positive: empty artifact list logs metrics only."""
        log_evaluation_results(sample_config, {"rmse": 1.0, "mae": 0.5}, [])

        mock_mlflow.start_run.assert_called_once_with(run_id="train-run-1")
        assert mock_mlflow.log_metric.call_count == 2
        mock_mlflow.log_artifact.assert_not_called()
