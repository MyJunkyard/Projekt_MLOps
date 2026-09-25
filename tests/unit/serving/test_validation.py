"""Unit tests for pure serving validation helpers."""

from unittest import mock

import pytest

from src.common.schema import ColumnRole, ColumnSpec, FeatureSchema
from src.serving.validation import (
    PredictionInputError,
    load_feature_schema,
    validate_feature_rows,
)


@pytest.fixture
def schema() -> FeatureSchema:
    return FeatureSchema(
        columns={
            "hour": ColumnSpec(name="hour", role=ColumnRole.FEATURE),
            "is_holiday": ColumnSpec(name="is_holiday", role=ColumnRole.FEATURE),
        }
    )


class TestValidateFeatureRows:
    def test_valid_rows_return_canonical_names(self, schema):
        assert validate_feature_rows(
            [{"hour": 1, "is_holiday": 0}],
            schema.feature_names(),
            1000,
        ) == ["hour", "is_holiday"]

    def test_empty_rows_rejected(self, schema):
        with pytest.raises(PredictionInputError, match="at least one row"):
            validate_feature_rows([], schema.feature_names(), 1000)

    def test_row_cap_rejected(self, schema):
        with pytest.raises(PredictionInputError, match="maximum"):
            validate_feature_rows(
                [{"hour": 1, "is_holiday": 0}] * 2,
                schema.feature_names(),
                1,
            )

    def test_missing_column_rejected(self, schema):
        with pytest.raises(PredictionInputError, match="missing"):
            validate_feature_rows([{"hour": 1}], schema.feature_names(), 1000)

    def test_extra_column_rejected(self, schema):
        with pytest.raises(PredictionInputError, match="unexpected"):
            validate_feature_rows(
                [{"hour": 1, "is_holiday": 0, "other": 1}],
                schema.feature_names(),
                1000,
            )

    @pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "1"])
    def test_non_finite_or_non_numeric_rejected(self, schema, value):
        with pytest.raises(PredictionInputError, match="finite number"):
            validate_feature_rows(
                [{"hour": value, "is_holiday": 0}],
                schema.feature_names(),
                1000,
            )

    def test_configured_range_is_inclusive_and_enforced(self, schema):
        assert validate_feature_rows(
            [{"hour": 23, "is_holiday": 1}],
            schema.feature_names(),
            1000,
            {"hour": [0, 23]},
        ) == ["hour", "is_holiday"]
        with pytest.raises(PredictionInputError, match="outside"):
            validate_feature_rows(
                [{"hour": 24, "is_holiday": 0}],
                schema.feature_names(),
                1000,
                {"hour": [0, 23]},
            )

    def test_optional_timestamp_must_be_timezone_aware(self, schema):
        with pytest.raises(PredictionInputError, match="timezone-aware"):
            validate_feature_rows(
                [{"hour": 1, "is_holiday": 0, "timestamp": "2024-01-01T00:00:00"}],
                schema.feature_names(),
                1000,
            )
        assert validate_feature_rows(
            [{"hour": 1, "is_holiday": 0, "timestamp": "2024-01-01T00:00:00Z"}],
            schema.feature_names(),
            1000,
        ) == ["hour", "is_holiday"]

    def test_schema_missing_skips_name_validation(self):
        assert validate_feature_rows([{"hour": 1}], None, 1000) is None


class TestLoadFeatureSchema:
    def test_loads_local_config_artifact(self, tmp_path, schema):
        artifact = tmp_path / "config" / "features_schema.json"
        schema.save(artifact)
        assert load_feature_schema("run-1", tmp_path) == schema

    def test_missing_local_artifact_returns_none(self, tmp_path):
        assert load_feature_schema("run-1", tmp_path) is None

    @mock.patch("src.serving.validation.mlflow.MlflowClient")
    def test_uses_resolved_run_id(self, client, tmp_path, schema):
        artifact = tmp_path / "features_schema.json"
        schema.save(artifact)
        client.return_value.download_artifacts.return_value = str(artifact)
        assert load_feature_schema("champion-run") == schema
        client.return_value.download_artifacts.assert_called_once_with(
            run_id="champion-run", artifact_path="config/features_schema.json"
        )
