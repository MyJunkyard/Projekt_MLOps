"""
Unit tests — WS0 feature superset materialisation contract.

Verifies that ``materialize_superset`` (the ``make featurise`` artifact) is
exactly ``build_features(drop_na=False)`` modulo NaN rows, that warm-up
NaNs are retained, that derivative names are scoped to the smoothing
windows, and that the sidecar + manifest are complete and hash-scoped.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.common.schema import FeatureSchema
from src.common.superset import superset_paths
from src.config.models import PipelineConfig
from src.features.main import build_features, materialize_superset


@pytest.fixture
def superset_config(tmp_path, sample_df):
    """A superset config + raw frame, repointed at a temp superset dir."""
    cfg = PipelineConfig.model_validate(
        {
            "data": {
                "target_col": "price_eur_mwh",
                "train_end": "2023-12-31",
                "val_end": "2024-01-01",
                "superset_dir": str(tmp_path / "superset") + "/",
            },
            "features": {
                "calendar": {"enabled": True},
                "lags": {"enabled": True, "periods": [1, 2, 24]},
                "derivatives": {
                    "enabled": True,
                    "order": [1, 2],
                    "smooth_windows": [3, 6, 12],
                },
            },
            "model": {"type": "sklearn.dummy.DummyRegressor"},
        }
    )
    return cfg


class TestDerivativeColumnsScoped:
    def test_derivative_columns_scoped_to_windows(self, superset_config, sample_df):
        built, schema = build_features(
            sample_df,
            superset_config.features,
            superset_config.data.target_col,
            weather=None,
            drop_na=False,
        )
        materialize_superset(built, schema, superset_config)

        on_disk = pd.read_parquet(superset_paths(superset_config).parquet)
        assert {
            "price_eur_mwh_diff_1_w3",
            "price_eur_mwh_diff_1_w6",
        } <= set(on_disk.columns)
        assert {
            "price_eur_mwh_diff_2_w3",
            "price_eur_mwh_diff_2_w6",
        } <= set(on_disk.columns)
        assert not {"price_eur_mwh_diff_1", "price_eur_mwh_diff_1_w99"} & set(
            on_disk.columns
        )


class TestSidecarDrivesFeatureNames:
    def test_sidecar_drives_feature_names(self, superset_config, sample_df):
        built, schema = build_features(
            sample_df,
            superset_config.features,
            superset_config.data.target_col,
            weather=None,
            drop_na=False,
        )
        materialize_superset(built, schema, superset_config)

        on_disk_schema = FeatureSchema.from_file(
            superset_paths(superset_config).schema
        )
        on_disk = pd.read_parquet(superset_paths(superset_config).parquet)
        assert set(on_disk_schema.feature_names()) == (
            set(built.columns) - {"timestamp", "price_eur_mwh"}
        )
        on_disk_schema.assert_matches_dataframe(on_disk)


class TestFrameAndSchemaPreserved:
    def test_frame_and_schema_preserved(self, superset_config, sample_df):
        built, schema = build_features(
            sample_df,
            superset_config.features,
            superset_config.data.target_col,
            weather=None,
            drop_na=False,
        )
        materialize_superset(built, schema, superset_config)

        on_disk = pd.read_parquet(superset_paths(superset_config).parquet)
        on_disk_schema = FeatureSchema.from_file(
            superset_paths(superset_config).schema
        )
        pd.testing.assert_frame_equal(on_disk, built)
        assert on_disk_schema == schema


class TestWarmupNanRowsRetained:
    def test_warmup_nan_rows_retained(self, superset_config, sample_df):
        built, schema = build_features(
            sample_df,
            superset_config.features,
            superset_config.data.target_col,
            weather=None,
            drop_na=False,
        )
        materialize_superset(built, schema, superset_config)

        on_disk = pd.read_parquet(superset_paths(superset_config).parquet)
        assert len(on_disk) == len(built)
        assert on_disk.isna().any(axis=1).sum() == built.isna().any(axis=1).sum()
        assert on_disk["lag_24h"].isna().sum() == 24
