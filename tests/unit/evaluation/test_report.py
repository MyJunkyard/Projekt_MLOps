"""Unit tests for evaluation/report.py — the cross-run comparison table (WS6).

The builder is a pure reader of the MLflow store; all MLflow is mocked so no
tracking server is needed in tier 1 (plan §8). These pin the filter string, the
run selection, the table shape, and the --dm reservation (D12).
"""

from unittest import mock

import pytest

from src.config.models import PipelineConfig
from src.evaluation.report import (
    REPORT_PARAM_COLUMNS,
    _build_filter_string,
    _format_row,
    build_report,
)


@pytest.fixture
def sample_config():
    """Minimal config with the two metric columns the table shows."""
    return PipelineConfig.model_validate(
        {
            "data": {
                "target_col": "price_eur_mwh",
                "train_end": "2023-12-31",
                "val_end": "2024-01-01",
            },
            "model": {"type": "sklearn.dummy.DummyRegressor"},
            "evaluation": {"metrics": ["rmse", "mae"]},
        }
    )


def _fake_run(run_id, name, params=None, metrics=None):
    """Build a MagicMock shaped like an MLflow run (data.tags/params/metrics)."""
    run = mock.MagicMock()
    run.info.run_id = run_id
    run.data.tags = {"mlflow.runName": name}
    run.data.params = params or {}
    run.data.metrics = metrics or {}
    return run


class TestBuildFilterString:
    """The series/exp_id → MLflow filter-string mapping."""

    def test_none(self):
        """Positive: no selectors → no filter (whole experiment)."""
        assert _build_filter_string(series=None, exp_id=None) is None

    def test_series_only(self):
        """Positive: series maps to the mlflow.series tag."""
        assert (
            _build_filter_string(series="B", exp_id=None)
            == "tags.`mlflow.series` = 'B'"
        )

    def test_exp_id_only(self):
        """Positive: exp_id maps to the mlflow.exp_id tag."""
        assert (
            _build_filter_string(series=None, exp_id="EXP-010")
            == "tags.`mlflow.exp_id` = 'EXP-010'"
        )

    def test_both_combined_with_and(self):
        """Positive: both selectors combine with 'and'."""
        result = _build_filter_string(series="B", exp_id="EXP-010")
        assert "mlflow.series" in result and "mlflow.exp_id" in result
        assert " and " in result


class TestFormatRow:
    """One run → a row of display strings, blank when a value is absent."""

    def test_params_and_metrics(self):
        """Positive: params and metrics land in the row; metrics are 4dp."""
        run = _fake_run(
            "r1",
            "xgboost-v1",
            params={"temporal_strategy": "single_step", "derivative_order": "1,2"},
            metrics={"rmse": 1.23456},
        )
        row = _format_row(run, ["rmse", "mae"])
        assert row["run_name"] == "xgboost-v1"
        assert row["temporal_strategy"] == "single_step"
        assert row["derivative_order"] == "1,2"
        assert row["rmse"] == "1.2346"
        assert row["mae"] == ""  # missing metric → blank, not 0

    def test_missing_params_blank(self):
        """Positive: a param the run did not log shows as blank."""
        run = _fake_run("r2", "baseline", params={}, metrics={})
        row = _format_row(run, ["rmse"])
        for col in REPORT_PARAM_COLUMNS:
            assert row[col] == ""


class TestBuildReport:
    """build_report: selection wiring, table shape, and the --dm reservation."""

    @mock.patch("src.evaluation.report.mlflow")
    def test_run_id_selects_single_run(self, mock_mlflow, sample_config):
        """Positive: an explicit run_id fetches exactly that run."""
        client = mock_mlflow.MlflowClient.return_value
        client.get_experiment_by_name.return_value = mock.MagicMock(
            experiment_id="exp-1"
        )
        client.get_run.return_value = _fake_run(
            "r-target", "arm-c", {"temporal_strategy": "direct"}, {"rmse": 2.0}
        )

        table = build_report(sample_config, run_id="r-target")

        client.get_run.assert_called_once_with("r-target")
        # search_runs is NOT used when a run_id is given.
        client.search_runs.assert_not_called()
        assert "arm-c" in table
        assert "direct" in table

    @mock.patch("src.evaluation.report.mlflow")
    def test_series_filter_builds_filter_string(self, mock_mlflow, sample_config):
        """Positive: series/exp_id reach search_runs as a filter string."""
        client = mock_mlflow.MlflowClient.return_value
        client.get_experiment_by_name.return_value = mock.MagicMock(
            experiment_id="exp-1"
        )
        client.search_runs.return_value = [
            _fake_run("r1", "arm-a", {}, {"rmse": 1.0})
        ]

        build_report(sample_config, series="B", exp_id="EXP-010")

        _, kwargs = client.search_runs.call_args
        assert "mlflow.series" in kwargs["filter_string"]
        assert "mlflow.exp_id" in kwargs["filter_string"]

    @mock.patch("src.evaluation.report.mlflow")
    def test_table_has_header_and_metric_columns(self, mock_mlflow, sample_config):
        """Positive: the table has a markdown header incl. the metric columns."""
        client = mock_mlflow.MlflowClient.return_value
        client.get_experiment_by_name.return_value = mock.MagicMock(
            experiment_id="exp-1"
        )
        client.search_runs.return_value = [
            _fake_run("r1", "arm-a", {}, {"rmse": 1.0, "mae": 0.5})
        ]

        table = build_report(sample_config)

        header = table.splitlines()[0]
        assert "run_name" in header
        assert "rmse" in header and "mae" in header
        # The §2 experiment params are columns too.
        assert "temporal_strategy" in header

    @mock.patch("src.evaluation.report.mlflow")
    def test_experiment_missing_returns_note(self, mock_mlflow, sample_config):
        """Negative: a missing experiment returns a note, does not raise."""
        client = mock_mlflow.MlflowClient.return_value
        client.get_experiment_by_name.return_value = None

        table = build_report(sample_config)

        assert "not found" in table

    def test_dm_reserved_raises_naming_ws7(self, sample_config):
        """Negative: --dm is reserved for WS7 and raises naming it (D12)."""
        with pytest.raises(NotImplementedError, match="WS7"):
            build_report(sample_config, dm=True)

