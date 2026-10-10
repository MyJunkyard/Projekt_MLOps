"""
Unit tests — common/superset.py (paths, manifest, materialisation hash).

Coverage: path derivation, manifest write/read round-trip, corrupt-manifest
handling, and the materialisation-hash scope (it must not depend on
``features.select``).
"""

from __future__ import annotations

import logging

import pandas as pd

from src.common.superset import (
    build_manifest,
    compute_materialisation_hash,
    read_manifest,
    superset_paths,
    write_manifest,
)
from src.config.models import PipelineConfig


def _base_data(tmp_path):
    """Shared minimal data config fragment used by the path/hash tests."""
    return {
        "data": {
            "target_col": "price_eur_mwh",
            "train_end": "2023-12-31",
            "val_end": "2024-01-01",
            "superset_dir": str(tmp_path / "superset") + "/",
        },
        "model": {"type": "sklearn.dummy.DummyRegressor"},
    }


def _minimal_cfg(tmp_path, resolution="hourly"):
    """A minimal PipelineConfig repointed at a temp superset dir."""
    return PipelineConfig.model_validate(
        {
            **_base_data(tmp_path),
            "temporal": {"resolution": resolution},
        }
    )


class TestSupersetPaths:
    """``superset_paths`` derives the three sibling files of one resolution."""

    def test_parquet_schema_manifest_siblings(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        paths = superset_paths(cfg)
        assert paths.parquet == tmp_path / "superset" / "features_hourly.parquet"
        assert paths.schema == tmp_path / "superset" / "features_hourly_schema.json"
        assert paths.manifest == tmp_path / "superset" / "manifest.json"

    def test_resolution_changes_filename(self, tmp_path):
        hourly = superset_paths(_minimal_cfg(tmp_path, "hourly"))
        daily = superset_paths(_minimal_cfg(tmp_path, "daily"))
        assert hourly.parquet.name == "features_hourly.parquet"
        assert daily.parquet.name == "features_daily.parquet"


class TestManifest:
    """Manifest write/read round-trip and corrupt-manifest behaviour."""

    def test_write_read_roundtrip(self, tmp_path):
        cfg = _minimal_cfg(tmp_path)
        paths = superset_paths(cfg)
        paths.ensure_dir()
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=10, freq="h", tz="UTC"
                ),
                "price_eur_mwh": list(range(10)),
            }
        )
        frame.to_parquet(paths.parquet, index=False)
        from src.common.schema import schema_from_dataframe

        schema_from_dataframe(frame, target_col="price_eur_mwh").save(paths.schema)
        manifest = build_manifest(paths, cfg, raw_csv_path=None)
        write_manifest(paths, manifest)

        assert paths.manifest.exists()
        loaded = read_manifest(paths)
        assert loaded is not None
        assert loaded.n_rows == len(frame)
        assert loaded.n_columns == frame.shape[1]
        assert loaded.resolution == "hourly"
        assert loaded.materialisation_hash
        assert loaded.params_hash
        assert loaded.schema_sha256
        assert loaded.superset_sha256
        assert loaded.git_commit

    def test_corrupt_manifest_returns_none_and_warns(self, tmp_path, caplog):
        cfg = _minimal_cfg(tmp_path)
        paths = superset_paths(cfg)
        paths.ensure_dir()
        paths.parquet.write_text("not a parquet", encoding="utf-8")
        paths.schema.write_text("{}", encoding="utf-8")
        paths.manifest.write_text("{not valid json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="src.common.superset"):
            assert read_manifest(paths) is None
        assert any("corrupt" in r.message.lower() for r in caplog.records)


class TestMaterialisationHash:
    """The materialisation hash is content-scoped, never affected by select."""

    def test_materialisation_hash_ignores_select(self, tmp_path):
        cfg = PipelineConfig.model_validate({**_base_data(tmp_path), "features": {}})
        cfg2 = PipelineConfig.model_validate(
            {
                **_base_data(tmp_path),
                "features": {"select": {"groups": ["calendar"]}},
            }
        )
        assert compute_materialisation_hash(cfg) == compute_materialisation_hash(cfg2)
