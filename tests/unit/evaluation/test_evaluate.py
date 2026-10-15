"""
Unit tests for the evaluation package — test feature loading, results
table, plots, residual breakdown, and MLflow result logging.
"""

import logging
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.common.schema import FeatureSchema
from src.common.selection import drop_selected_nan_rows, open_superset
from src.config.models import EvaluationConfig
from src.evaluation.plots import (
    plot_actual_vs_predicted,
    plot_actual_vs_predicted_window,
    plot_residual_breakdown,
    plot_residuals_over_time,
    residual_breakdown,
)
from src.evaluation.reporting import (
    get_model_run_id,
    load_test_features,
    log_evaluation_results,
    log_results_table,
    resolve_evaluate_columns,
)
from src.training.loader import load_superset_features


class TestLoadTestFeatures:
    @mock.patch("src.evaluation.reporting.get_model_run_id", return_value=None)
    def test_returns_x_and_y(self, mock_run_id, superset_dir, sample_config):
        """Positive: returns X_test and y_test arrays."""
        frame, schema = open_superset(sample_config)
        feature_cols, _ = resolve_evaluate_columns(sample_config, schema)
        X_test, y_test = load_test_features(
            frame, feature_cols, sample_config.data.target_col
        )
        assert X_test.shape[0] == y_test.shape[0]
        assert X_test.shape[0] > 0
        assert X_test.shape[1] == len(feature_cols)

    def test_missing_file_raises(self, tmp_path, sample_config):
        """Negative: a frame without all resolved features raises ValueError."""
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=10, freq="h", tz="UTC"
                ),
                "price_eur_mwh": np.linspace(40, 60, 10),
            }
        )
        with pytest.raises(ValueError, match="missing"):
            load_test_features(
                frame,
                [
                    "timestamp",
                    "price_eur_mwh",
                    "no_such_feature",
                ],
                sample_config.data.target_col,
            )

    @mock.patch("src.evaluation.reporting.get_model_run_id", return_value=None)
    def test_excludes_meta_columns_via_sidecar(
        self, mock_run_id, superset_dir, sample_config, tmp_path
    ):
        """Sidecar-driven selection excludes META columns (is_imputed)."""
        from src.common.schema import schema_from_dataframe

        frame, schema = open_superset(sample_config)
        feature_cols, _ = resolve_evaluate_columns(sample_config, schema)
        df_clean = drop_selected_nan_rows(
            frame, feature_cols, sample_config.data.target_col
        )

        # The real featurise schema marks known meta columns as META:
        # they are excluded from ``feature_names()`` and never reach X.
        df_meta = df_clean.copy()
        df_meta["is_imputed"] = False
        meta_schema = schema_from_dataframe(
            df_meta,
            target_col=sample_config.data.target_col,
            meta_cols=("is_imputed",),
        )
        sidecar_dir = tmp_path / "with_sidecar"
        sidecar_dir.mkdir()
        meta_schema.save(sidecar_dir / "features_schema.json")
        loaded_meta_schema = FeatureSchema.from_file(
            sidecar_dir / "features_schema.json"
        )

        meta_feature_cols, _ = resolve_evaluate_columns(
            sample_config, loaded_meta_schema
        )
        X_test, y_test = load_test_features(
            df_meta, meta_feature_cols, sample_config.data.target_col
        )
        assert "is_imputed" not in meta_feature_cols
        assert "is_imputed" not in loaded_meta_schema.feature_names()
        assert X_test.shape[1] == len(meta_feature_cols)
        # Identical matrix to what training would feed the model: same
        # clean frame, same canonical column order (the splits are
        # row-slices of it in chronological order).
        X_splits, _, _, _, _ = load_superset_features(sample_config)
        np.testing.assert_array_equal(X_test, np.concatenate(X_splits))


class TestLogResultsTable:
    def test_logs_metrics(self, caplog):
        """Positive: logs metrics at INFO level."""
        with caplog.at_level(logging.INFO, logger="src.evaluation.reporting"):
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
        evaluation = EvaluationConfig(residual_breakdown=["hour"])

        result = residual_breakdown(df, y_true, y_pred, evaluation, quantiles=0)
        assert "grouping" in result.columns
        assert "category" in result.columns
        assert "mean_abs_residual" in result.columns
        # quantiles=0 disables the price-quantile grouping; direction is always
        # added, so filter to the categorical grouping for the count.
        hour_rows = result[result["grouping"] == "hour"]
        assert len(hour_rows) == 3  # hours 0, 1, 2
        # Hour 0: |10-15| + |20-15| / 2 = (5+5)/2 = 5
        hour0 = hour_rows[hour_rows["category"] == 0]
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
        evaluation = EvaluationConfig(residual_breakdown=["is_holiday"])

        result = residual_breakdown(df, y_true, y_pred, evaluation, quantiles=0)
        holiday_rows = result[result["grouping"] == "is_holiday"]
        assert len(holiday_rows) == 2
        # Holidays have mean residual of (5+5)/2 = 5
        holiday_row = holiday_rows[holiday_rows["category"] == True]  # noqa: E712
        assert holiday_row["mean_abs_residual"].iloc[0] == pytest.approx(5.0)

    def test_returns_dataframe(self):
        """Positive: returns a DataFrame."""
        df = pd.DataFrame({"hour": [0, 1]})
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        evaluation = EvaluationConfig(residual_breakdown=["hour"])

        result = residual_breakdown(df, y_true, y_pred, evaluation)
        assert isinstance(result, pd.DataFrame)

    def test_price_quantile_grouping(self):
        """Positive (WS4): quantiles=4 bins the actual price into 4 bins."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        y_pred = y_true + 1.0
        evaluation = EvaluationConfig(residual_breakdown=[])
        df = pd.DataFrame({"hour": np.arange(8) % 24})

        result = residual_breakdown(df, y_true, y_pred, evaluation, quantiles=4)
        q_rows = result[result["grouping"] == "price_quantile"]
        assert len(q_rows) == 4
        # 8 evenly-spread values into 4 bins → 2 rows each.
        assert (q_rows["count"] == 2).all()

    def test_quantiles_zero_disables_grouping(self):
        """Negative (WS4): quantiles=0 adds no price_quantile rows."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = y_true
        evaluation = EvaluationConfig(residual_breakdown=[])
        df = pd.DataFrame({"hour": np.arange(4)})

        result = residual_breakdown(df, y_true, y_pred, evaluation, quantiles=0)
        assert not (result["grouping"] == "price_quantile").any()
        # Direction is still always present.
        assert (result["grouping"] == "direction").any()

    def test_direction_classification(self):
        """Positive (WS4): rising/falling/flat from the actual's own change."""
        # diffs: nan, +1, +1, 0, -1  → rising(2), flat(1), falling(1)
        y_true = np.array([10.0, 11.0, 12.0, 12.0, 11.0])
        y_pred = y_true
        evaluation = EvaluationConfig(residual_breakdown=[])
        df = pd.DataFrame({"hour": np.arange(5)})

        result = residual_breakdown(df, y_true, y_pred, evaluation, quantiles=0)
        direction = result[result["grouping"] == "direction"].set_index("category")
        assert direction.loc["rising", "count"] == 2
        assert direction.loc["flat", "count"] == 1
        assert direction.loc["falling", "count"] == 1
        # The first row (NaN direction) is excluded from every direction group.
        assert direction["count"].sum() == 4


class TestResidualAnalysisPlots:
    """WS4 step 1 — the three new residual-analysis plot functions."""

    def test_residuals_over_time_creates_png(self, tmp_path):
        """Positive: writes a non-empty PNG for a multi-row series."""
        out = tmp_path / "rot.png"
        ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        residuals = np.linspace(-5, 5, 48)
        result = plot_residuals_over_time(ts, residuals, str(out))
        assert result == str(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_residuals_over_time_single_row_no_raise(self, tmp_path):
        """Edge: a single-row input renders without raising (rolling clamps)."""
        out = tmp_path / "rot1.png"
        ts = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
        result = plot_residuals_over_time(ts, np.array([2.0]), str(out))
        assert result == str(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_window_selects_rows_and_creates_png(self, tmp_path):
        """Positive: a window overlapping the data renders actual-vs-predicted."""
        out = tmp_path / "win.png"
        ts = pd.date_range("2024-03-04", periods=168, freq="h", tz="UTC")
        y_true = np.linspace(40, 60, 168)
        y_pred = y_true + 1.0
        result = plot_actual_vs_predicted_window(
            ts, y_true, y_pred, "2024-03-04", "2024-03-11", str(out)
        )
        assert result == str(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_window_out_of_range_returns_none(self, tmp_path):
        """Negative: a window selecting no rows returns None (no file written)."""
        out = tmp_path / "empty.png"
        ts = pd.date_range("2024-03-04", periods=24, freq="h", tz="UTC")
        y_true = np.linspace(40, 60, 24)
        y_pred = y_true
        result = plot_actual_vs_predicted_window(
            ts, y_true, y_pred, "2030-01-01", "2030-01-08", str(out)
        )
        assert result is None
        assert not out.exists()

    def test_residual_breakdown_facets_groupings(self, tmp_path):
        """Positive: one subplot per grouping column; non-empty PNG."""
        out = tmp_path / "breakdown.png"
        df = pd.DataFrame(
            {
                "grouping": ["hour", "hour", "day_of_week", "day_of_week"],
                "category": [0, 1, 0, 1],
                "mean_abs_residual": [1.0, 2.0, 3.0, 4.0],
                "count": [10, 10, 10, 10],
            }
        )
        result = plot_residual_breakdown(df, str(out))
        assert result == str(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_residual_breakdown_empty_frame_creates_png(self, tmp_path):
        """Edge: an empty breakdown frame still renders a valid (empty) PNG."""
        out = tmp_path / "empty_breakdown.png"
        df = pd.DataFrame(
            {"grouping": [], "category": [], "mean_abs_residual": [], "count": []}
        )
        result = plot_residual_breakdown(df, str(out))
        assert result == str(out)
        assert out.exists()
        assert out.stat().st_size > 0


class TestGetModelRunId:
    @mock.patch("src.evaluation.reporting.mlflow")
    def test_returns_run_id_from_alias(self, mock_mlflow, sample_config):
        """Positive: resolves the training run ID from the champion alias."""
        mock_client = mock_mlflow.MlflowClient.return_value
        mock_client.get_model_version_by_alias.return_value.run_id = "train-run-1"

        run_id = get_model_run_id(sample_config)

        assert run_id == "train-run-1"
        mock_client.get_model_version_by_alias.assert_called_once_with(
            sample_config.mlflow.model_name,
            sample_config.serving.model_alias,
        )

    @mock.patch("src.evaluation.reporting.mlflow")
    def test_returns_none_on_registry_error(self, mock_mlflow, sample_config):
        """Negative: registry failure returns None instead of raising."""
        client = mock_mlflow.MlflowClient.return_value
        client.get_model_version_by_alias.side_effect = RuntimeError("registry down")

        assert get_model_run_id(sample_config) is None


class TestLogEvaluationResults:
    @mock.patch(
        "src.evaluation.reporting.get_model_run_id", return_value="train-run-1"
    )
    @mock.patch("src.evaluation.reporting.mlflow")
    def test_logs_to_training_run(self, mock_mlflow, mock_run_id, sample_config):
        """Positive: metrics and artifacts go to the training run."""
        log_evaluation_results(
            sample_config, {"rmse": 1.0}, ["plot.png", "breakdown.csv"]
        )

        mock_mlflow.start_run.assert_called_once_with(run_id="train-run-1")
        mock_mlflow.log_metric.assert_any_call("rmse", 1.0)
        logged = [call[0][0] for call in mock_mlflow.log_artifact.call_args_list]
        assert logged == ["plot.png", "breakdown.csv"]

    @mock.patch(
        "src.evaluation.reporting.get_model_run_id", return_value=None
    )
    @mock.patch("src.evaluation.reporting.mlflow")
    def test_falls_back_to_named_run(self, mock_mlflow, mock_run_id, sample_config):
        """Negative: unresolved training run falls back to a named run."""
        log_evaluation_results(sample_config, {"rmse": 1.0}, [])

        mock_mlflow.start_run.assert_called_once_with(run_name="evaluation")
        mock_mlflow.log_metric.assert_any_call("rmse", 1.0)
        mock_mlflow.log_artifact.assert_not_called()

    @mock.patch(
        "src.evaluation.reporting.get_model_run_id", return_value="train-run-1"
    )
    @mock.patch("src.evaluation.reporting.mlflow")
    def test_empty_artifacts_ok(self, mock_mlflow, mock_run_id, sample_config):
        """Positive: empty artifact list logs metrics only."""
        log_evaluation_results(sample_config, {"rmse": 1.0, "mae": 0.5}, [])

        mock_mlflow.start_run.assert_called_once_with(run_id="train-run-1")
        assert mock_mlflow.log_metric.call_count == 2
        mock_mlflow.log_artifact.assert_not_called()
