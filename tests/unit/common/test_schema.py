"""
Unit tests for the column metadata registry (``common/schema.py``).

Covers ``ColumnSpec``/``FeatureSchema`` construction, lookups
(``feature_names``/``select``), the drift guard
(``assert_matches_dataframe``), persistence round-trips, and the
graceful ``load_schema_or_none`` behavior.
"""

import pandas as pd
import pytest

from src.common.schema import (
    ColumnRole,
    ColumnSpec,
    FeatureSchema,
    load_schema_or_none,
    schema_from_dataframe,
)


def _spec(name: str, role: ColumnRole = ColumnRole.FEATURE, **kwargs) -> ColumnSpec:
    return ColumnSpec(name=name, role=role, **kwargs)


def _schema() -> FeatureSchema:
    return FeatureSchema(
        columns={
            "timestamp": _spec("timestamp", ColumnRole.IDENTIFIER, group="base"),
            "price_eur_mwh": _spec("price_eur_mwh", ColumnRole.TARGET, group="base"),
            "hour": _spec("hour", group="calendar"),
            "load_mw_lag1h": _spec(
                "load_mw_lag1h",
                group="availability_lag",
                availability_lag_hours=1,
                derived_from=["load_mw"],
            ),
        }
    )


class TestColumnSpec:
    def test_default_availability_lag_is_zero(self):
        assert ColumnSpec(name="x", role=ColumnRole.FEATURE).availability_lag_hours == 0

    def test_negative_availability_lag_rejected(self):
        with pytest.raises(ValueError, match="ge=0|greater than or equal"):
            ColumnSpec(name="x", role=ColumnRole.FEATURE, availability_lag_hours=-1)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValueError):
            ColumnSpec.model_validate(
                {"name": "x", "role": "feature", "typo_field": 1}
            )


class TestFeatureSchema:
    def test_names_preserve_order(self):
        assert _schema().names() == [
            "timestamp",
            "price_eur_mwh",
            "hour",
            "load_mw_lag1h",
        ]

    def test_feature_names_excludes_non_features(self):
        assert _schema().feature_names() == ["hour", "load_mw_lag1h"]

    def test_select_by_group(self):
        assert _schema().select(group="availability_lag") == ["load_mw_lag1h"]

    def test_select_by_role(self):
        assert _schema().select(role=ColumnRole.TARGET) == ["price_eur_mwh"]

    def test_select_combined_role_and_group(self):
        assert _schema().select(role=ColumnRole.FEATURE, group="calendar") == ["hour"]

    def test_select_no_match_returns_empty(self):
        assert _schema().select(group="nope") == []


class TestDriftGuard:
    def test_matching_frame_passes(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
                "price_eur_mwh": [1.0, 2.0, 3.0],
                "hour": [0, 1, 2],
                "load_mw_lag1h": [0.0, 1.0, 2.0],
            }
        )
        _schema().assert_matches_dataframe(df)  # must not raise

    def test_missing_column_raises(self):
        df = pd.DataFrame({"timestamp": [1], "price_eur_mwh": [1.0], "hour": [0]})
        with pytest.raises(ValueError, match="load_mw_lag1h"):
            _schema().assert_matches_dataframe(df)

    def test_undeclared_column_raises(self):
        df = pd.DataFrame({"timestamp": [1], "price_eur_mwh": [1.0], "extra": [0]})
        with pytest.raises(ValueError, match="extra"):
            _schema().assert_matches_dataframe(df)


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        schema = _schema()
        path = schema.save(tmp_path / "schema.json")
        loaded = FeatureSchema.from_file(path)
        assert loaded == schema

    def test_load_schema_or_none_missing_file(self, tmp_path):
        assert load_schema_or_none(tmp_path / "absent.json") is None

    def test_load_schema_or_none_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json {", encoding="utf-8")
        assert load_schema_or_none(path) is None


class TestSchemaFromDataframe:
    def test_covers_all_columns(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
                "price_eur_mwh": [1.0, 2.0],
                "is_imputed": [False, True],
                "load_mw": [10.0, 11.0],
            }
        )
        schema = schema_from_dataframe(df, target_col="price_eur_mwh")
        schema.assert_matches_dataframe(df)
        assert schema.select(role=ColumnRole.TARGET) == ["price_eur_mwh"]
        assert schema.select(role=ColumnRole.META) == ["is_imputed"]
        assert schema.feature_names() == ["load_mw"]
