"""
Unit tests — training selection against a materialised superset.

Covers groups, explicit columns, the ``lags`` group alias, canonical
(schema) ordering, actionable errors for missing groups / missing superset,
per-subset NaN drop, and stale-hash WARNING.
"""

from __future__ import annotations

import logging

import pytest

from src.common.superset import (
    build_manifest,
    read_manifest,
    superset_paths,
    write_manifest,
)
from src.config.models import FeatureSelection, PipelineConfig
from src.features.main import build_features
from src.training.loader import (
    _warn_if_stale,
    load_superset_features,
    open_superset,
    resolve_selection,
)


def _base_data(tmp_path):
    """Shared minimal data config fragment used by the selection tests."""
    return {
        "data": {
            "target_col": "price_eur_mwh",
            "train_end": "2023-12-31",
            "val_end": "2024-01-01",
            "superset_dir": str(tmp_path / "superset") + "/",
        },
        "model": {"type": "sklearn.dummy.DummyRegressor"},
    }


def _make_cfg(tmp_path, features):
    """Build a minimal PipelineConfig with the given feature toggles."""
    return PipelineConfig.model_validate(
        {
            **_base_data(tmp_path),
            "features": features,
        }
    )


def _materialise_into(cfg, df, directory):
    """Featurise-equivalent: write the three superset artifacts for cfg."""
    directory.mkdir(parents=True, exist_ok=True)
    cfg.data.superset_dir = str(directory) + "/"
    paths = superset_paths(cfg)
    built, schema = build_features(
        df, cfg.features, cfg.data.target_col, weather=None, drop_na=False
    )
    built.to_parquet(paths.parquet, index=False)
    schema.save(paths.schema)
    write_manifest(paths, build_manifest(paths, cfg, raw_csv_path=None))
    return built, schema, paths


@pytest.fixture
def superset_config(tmp_path, sample_df):
    """A superset-config + frame materialised into a temp directory."""
    cfg = _make_cfg(
        tmp_path,
        {
            "calendar": {"enabled": True},
            "lags": {"enabled": True, "periods": [1, 2, 24]},
        },
    )
    _materialise_into(cfg, sample_df, tmp_path / "superset")
    return cfg


class TestResolveSelection:
    """``resolve_selection`` resolves the superset schema to feature columns."""

    def test_none_selects_every_feature(self, superset_config):
        cfg = superset_config
        _, schema = open_superset(cfg)
        names = resolve_selection(schema, None)
        assert names == schema.feature_names()
        assert "timestamp" not in names
        assert "price_eur_mwh" not in names

    def test_groups(self, superset_config):
        cfg = superset_config
        _, schema = open_superset(cfg)
        names = resolve_selection(
            schema, FeatureSelection(groups=["calendar", "lags"])
        )
        assert set(names) == {
            "hour",
            "day_of_week",
            "month",
            "week_of_year",
            "is_holiday",
            "is_workday",
            "days_to_next_holiday",
            "days_since_last_holiday",
            "lag_1h",
            "lag_2h",
            "lag_24h",
            "rolling_mean_24h",
            "rolling_std_24h",
            "rolling_mean_168h",
            "rolling_std_168h",
        }

    def test_columns(self, superset_config):
        cfg = superset_config
        _, schema = open_superset(cfg)
        names = resolve_selection(
            schema, FeatureSelection(columns=["lag_24h", "hour"])
        )
        assert names == ["hour", "lag_24h"]

    def test_groups_and_columns(self, superset_config):
        cfg = superset_config
        _, schema = open_superset(cfg)
        names = resolve_selection(
            schema, FeatureSelection(groups=["calendar"], columns=["lag_24h"])
        )
        assert set(names) == {
            "hour",
            "day_of_week",
            "month",
            "week_of_year",
            "is_holiday",
            "is_workday",
            "days_to_next_holiday",
            "days_since_last_holiday",
            "lag_24h",
        }

    def test_lags_alias_expands_to_lag_and_rolling(self, superset_config):
        cfg = superset_config
        _, schema = open_superset(cfg)
        names = resolve_selection(
            schema, FeatureSelection(groups=["lags"])
        )
        assert set(names) == {
            "lag_1h",
            "lag_2h",
            "lag_24h",
            "rolling_mean_24h",
            "rolling_std_24h",
            "rolling_mean_168h",
            "rolling_std_168h",
        }

    def test_groups_are_in_canonical_schema_order(self, superset_config):
        cfg = superset_config
        _, schema = open_superset(cfg)
        names = resolve_selection(
            schema, FeatureSelection(groups=["lags", "calendar"])
        )
        assert names == [
            "hour",
            "day_of_week",
            "month",
            "week_of_year",
            "is_holiday",
            "is_workday",
            "days_to_next_holiday",
            "days_since_last_holiday",
            "lag_1h",
            "lag_2h",
            "lag_24h",
            "rolling_mean_24h",
            "rolling_std_24h",
            "rolling_mean_168h",
            "rolling_std_168h",
        ]


class TestResolveSelectionErrors:
    def test_unknown_group_raises(self, superset_config):
        cfg = superset_config
        _, schema = open_superset(cfg)
        with pytest.raises(ValueError, match="weather"):
            resolve_selection(schema, FeatureSelection(groups=["weather"]))

    def test_group_absent_from_superset_raises_actionable(self, tmp_path, sample_df):
        cfg = _make_cfg(
            tmp_path,
            {
                "calendar": {"enabled": True},
                "lags": {"enabled": True, "periods": [1, 2, 24]},
            },
        )
        _materialise_into(cfg, sample_df, tmp_path / "superset")
        _, schema = open_superset(cfg)
        with pytest.raises(ValueError, match="weather"):
            resolve_selection(schema, FeatureSelection(groups=["weather"]))


class TestPerSubsetNaN:
    def test_per_subset_nan_drop(self, sample_df, tmp_path):
        cfg_calendar = _make_cfg(
            tmp_path,
            {
                "calendar": {"enabled": True},
            },
        )
        cfg_lags = _make_cfg(
            tmp_path,
            {
                "lags": {"enabled": True, "periods": [1, 2, 24]},
            },
        )
        cal_dir = tmp_path / "calendar"
        lags_dir = tmp_path / "lags"
        frame = sample_df.copy()

        _materialise_into(cfg_calendar, frame, cal_dir)
        _materialise_into(cfg_lags, frame, lags_dir)

        X_cal, _, _, _, clean_cal = load_superset_features(cfg_calendar)
        X_lags, _, _, _, clean_lags = load_superset_features(cfg_lags)

        assert len(clean_cal) == 72
        assert len(clean_lags) == 48
        assert X_cal[0].shape[0] == 24
        assert X_cal[1].shape[0] == 24
        assert X_cal[2].shape[0] == 24
        assert X_lags[0].shape[0] == 0
        assert X_lags[1].shape[0] == 24
        assert X_lags[2].shape[0] == 24


class TestMissingArtifacts:
    def test_missing_superset_raises_actionable(self, tmp_path):
        cfg = _make_cfg(tmp_path, {"calendar": {"enabled": True}})
        cfg.data.superset_dir = str(tmp_path / "absent") + "/"
        with pytest.raises(FileNotFoundError, match="featurise"):
            load_superset_features(cfg)

    def test_missing_schema_raises_actionable(self, tmp_path, sample_df):
        cfg = _make_cfg(tmp_path, {"calendar": {"enabled": True}})
        directory = tmp_path / "superset"
        _materialise_into(cfg, sample_df, directory)
        (directory / "features_hourly_schema.json").unlink()
        with pytest.raises(FileNotFoundError, match="schema sidecar"):
            load_superset_features(cfg)

    def test_missing_manifest_raises_actionable(self, tmp_path, sample_df):
        cfg = _make_cfg(tmp_path, {"calendar": {"enabled": True}})
        directory = tmp_path / "superset"
        _materialise_into(cfg, sample_df, directory)
        (directory / "manifest.json").unlink()
        with pytest.raises(FileNotFoundError, match="manifest"):
            load_superset_features(cfg)


class TestStaleHash:
    def test_stale_hash_warns(self, tmp_path, caplog, sample_df):
        cfg = _make_cfg(tmp_path, {"calendar": {"enabled": True}})
        directory = tmp_path / "superset"
        _materialise_into(cfg, sample_df, directory)

        paths = superset_paths(cfg)
        manifest = build_manifest(paths, cfg, raw_csv_path=None)
        manifest.materialisation_hash = "deadbeefdeadbeef"
        write_manifest(paths, manifest)

        with caplog.at_level(logging.WARNING, logger="src.training.loader"):
            _warn_if_stale(cfg, read_manifest(paths))
        assert any("STALE" in r.message for r in caplog.records)
