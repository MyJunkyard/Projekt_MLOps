"""Unit tests for training/registry.py — the WS6 experiment-hygiene logging.

These pin the flat experiment params, the run_tags application, the stage tag,
the superset-manifest provenance, per-split row counts, the param-collision
guard, the superset/feature group params, and the evaluate run-id override.
All MLflow is mocked — no tracking server in tier 1 (plan §8).
"""

from unittest import mock

import pytest

from src.common.schema import ColumnRole, ColumnSpec, FeatureSchema
from src.config.models import FeatureSelection
from src.training.registry import _join, log_to_mlflow


def _run_ctx(mock_mlflow, run_id="run-1", registered=None):
    """Wire a mocked ``mlflow.start_run`` context manager returning ``run_id``.

    Mirrors the setup in ``test_train.py`` so both files read the same way.
    """
    mock_run = mock.MagicMock()
    mock_run.info.run_id = run_id
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.MlflowClient.return_value = mock.MagicMock()
    return mock_run


def _logged_params(mock_mlflow) -> dict:
    """Collect every ``log_param(key, value)`` call into a dict (last wins)."""
    return {
        call.args[0]: call.args[1]
        for call in mock_mlflow.log_param.call_args_list
    }


def _subset_schema() -> FeatureSchema:
    """A small subset schema: calendar + lags FEATURE columns only.

    Mirrors what ``loader.load_superset_features`` logs: ``schema.subset(
    feature_cols)`` where ``feature_cols`` are the FEATURE-role columns — so
    the identifier/target are NOT in the subset schema.
    """
    return FeatureSchema(
        columns={
            "hour": ColumnSpec(name="hour", role=ColumnRole.FEATURE, group="calendar"),
            "lag_1h": ColumnSpec(name="lag_1h", role=ColumnRole.FEATURE, group="lag"),
        }
    )


class TestExperimentParams:
    """The §2 flat params: temporal strategy, derivatives, selection echo."""

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_temporal_and_derivative_params(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Positive: temporal/derivative params are logged with joined lists."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        params = _logged_params(mock_mlflow)
        assert params["temporal_strategy"] == sample_config.temporal.strategy
        # D2: lists are comma-joined, not str(list).
        assert params["derivative_order"] == "1,2"
        assert params["derivative_smooth_windows"] == "3,6,12"
        assert params["weather_derivative_order"] == "1"
        assert params["weather_derivative_smooth_windows"] == "3"

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_weather_params_present_when_disabled(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Positive: weather-derivative params logged even when block disabled."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)
        assert sample_config.features.weather.derivatives.enabled is False

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        params = _logged_params(mock_mlflow)
        assert params["weather_derivative_order"] == "1"
        assert params["weather_derivative_smooth_windows"] == "3"

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_select_none_reads_all(self, mock_mlflow, mock_log_model, sample_config):
        """Positive: select_groups='all' and select_columns_n=0 when no selection."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)
        assert sample_config.features.select is None

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        params = _logged_params(mock_mlflow)
        assert params["select_groups"] == "all"
        assert params["select_columns_n"] == 0

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_select_groups_and_columns(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Positive: a groups selection echoes joined groups; '-' when empty."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)
        sample_config.features.select = FeatureSelection(groups=["calendar", "lags"])

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        params = _logged_params(mock_mlflow)
        assert params["select_groups"] == "calendar,lags"
        assert params["select_columns_n"] == 0  # no explicit column list

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_select_columns_only_reads_dash(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Positive: a columns-only selection echoes '-' groups and the count."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)
        sample_config.features.select = FeatureSelection(columns=["hour", "lag_1h"])

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        params = _logged_params(mock_mlflow)
        assert params["select_groups"] == "-"
        assert params["select_columns_n"] == 2


class TestJoin:
    """The D2 list-serialisation helper in isolation."""

    def test_join_ints(self):
        """Positive: ints are comma-joined without brackets."""
        assert _join([1, 2]) == "1,2"
        assert _join([3, 6, 12]) == "3,6,12"

    def test_join_empty(self):
        """Positive: an empty list joins to the empty string."""
        assert _join([]) == ""


class TestRunTags:
    """``mlflow.run_tags`` are applied as tags (D4)."""

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_run_tags_applied(self, mock_mlflow, mock_log_model, sample_config):
        """Positive: configured run_tags reach set_tag as strings."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)
        sample_config.mlflow.run_tags = {"exp_id": "EXP-010", "series": "B"}

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        mock_mlflow.set_tag.assert_any_call("exp_id", "EXP-010")
        mock_mlflow.set_tag.assert_any_call("series", "B")

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_empty_run_tags_logs_nothing(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Positive: the empty default sets no run_tags tag."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)
        assert sample_config.mlflow.run_tags == {}

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        tagged = {c.args[0] for c in mock_mlflow.set_tag.call_args_list}
        assert "exp_id" not in tagged and "series" not in tagged


class TestStageTag:
    """``stage`` is written by code as the module constant (D9)."""

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_stage_is_four(self, mock_mlflow, mock_log_model, sample_config):
        """Positive: every Stage-4 run carries stage='4'."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        mock_mlflow.set_tag.assert_any_call("stage", "4")


class TestSupersetManifest:
    """Superset provenance: artifact + tags, with graceful absence (D5)."""

    @mock.patch("src.training.registry.read_manifest")
    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_manifest_logged_under_config_superset(
        self, mock_mlflow, mock_log_model, mock_read_manifest, sample_config
    ):
        """Positive: the superset manifest lands at config/superset/manifest.json."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)
        mock_read_manifest.return_value = mock.MagicMock(
            materialisation_hash="deadbeefcafe"
        )

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        # Artifact nested so it cannot overwrite the raw config/manifest.json.
        superset_calls = [
            c
            for c in mock_mlflow.log_artifact.call_args_list
            if c.kwargs.get("artifact_path") == "config/superset"
        ]
        assert superset_calls, "superset manifest artifact not logged"
        mock_mlflow.set_tag.assert_any_call(
            "superset_materialisation_hash", "deadbeefcafe"
        )
        assert any(
            c.args[0] == "superset_path" for c in mock_mlflow.set_tag.call_args_list
        )

    @mock.patch("src.training.registry.read_manifest", return_value=None)
    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_missing_manifest_warns_not_raises(
        self, mock_mlflow, mock_log_model, mock_read_manifest, sample_config
    ):
        """Negative: a missing manifest only warns; the run still logs params."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)

        # Must not raise even though provenance is absent.
        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        # No superset provenance tags written.
        tagged = {c.args[0] for c in mock_mlflow.set_tag.call_args_list}
        assert "superset_materialisation_hash" not in tagged
        assert "superset_path" not in tagged
        # The rest of the run is unaffected.
        assert (
            _logged_params(mock_mlflow)["temporal_strategy"]
            == sample_config.temporal.strategy
        )



class TestRowCounts:
    """Per-split row counts are params, only when supplied (D6)."""

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_row_counts_logged(self, mock_mlflow, mock_log_model, sample_config):
        """Positive: row_counts become n_rows_train/_val/_test params."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)

        log_to_mlflow(
            mock.MagicMock(),
            {"rmse": 1.0},
            sample_config,
            row_counts={"train": 100, "val": 20, "test": 30},
        )

        params = _logged_params(mock_mlflow)
        assert params["n_rows_train"] == 100
        assert params["n_rows_val"] == 20
        assert params["n_rows_test"] == 30

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_absent_row_counts_absent_params(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Negative: no row_counts argument → no n_rows_* params (not zero)."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        params = _logged_params(mock_mlflow)
        assert "n_rows_train" not in params
        assert "n_rows_val" not in params
        assert "n_rows_test" not in params


class TestParamGuard:
    """The _log_param_once collision guard (D1)."""

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_collision_raises_loudly(self, mock_mlflow, mock_log_model, sample_config):
        """Negative: a model.params key colliding with a pipeline param raises.

        ``temporal_strategy`` is logged by the pipeline; model.params must not
        reuse it. The guard fails here, before the run is committed, instead of
        deep inside MLflow's "Changing param values" error.
        """
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)
        sample_config.model.params = {"temporal_strategy": "forged"}

        with pytest.raises(ValueError, match="temporal_strategy"):
            log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)


class TestGroupParams:
    """superset_groups (always) and feature_groups/n_features (schema only)."""

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_superset_groups_from_enabled_flags(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Positive: superset_groups reflects the enabled feature blocks."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)
        # sample_config has calendar + lags enabled only.
        assert sample_config.features.calendar.enabled is True
        assert sample_config.features.lags.enabled is True
        assert sample_config.features.weather.enabled is False

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        params = _logged_params(mock_mlflow)
        assert params["superset_groups"] == "calendar,lags"

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_feature_groups_only_with_schema(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Positive: feature_groups/n_features come from the passed subset schema."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)

        log_to_mlflow(
            mock.MagicMock(),
            {"rmse": 1.0},
            sample_config,
            feature_schema=_subset_schema(),
        )

        params = _logged_params(mock_mlflow)
        assert params["n_features"] == 2  # hour + lag_1h
        assert params["feature_groups"] == "calendar,lag"

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_baselines_carry_neither_group_param(
        self, mock_mlflow, mock_log_model, sample_config
    ):
        """Negative: no feature_schema → no feature_groups/n_features (baselines)."""
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)
        _run_ctx(mock_mlflow)

        log_to_mlflow(mock.MagicMock(), {"rmse": 1.0}, sample_config)

        params = _logged_params(mock_mlflow)
        assert "n_features" not in params
        assert "feature_groups" not in params
        # But the config echo is still there.
        assert "superset_groups" in params

