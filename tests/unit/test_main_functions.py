"""
Unit tests for the main() CLI entry points of each pipeline module.

These exercise the orchestration functions by mocking load_config and
external services, so no real config or MLflow server is needed.
"""

from unittest import mock

import numpy as np
import pandas as pd

import src.evaluation.main as evaluate_mod
import src.features.main as featurise_mod
import src.ingestion.main as ingest_mod
import src.training.main as train_mod
from src.config.models import PipelineConfig


class TestIngestMain:
    @mock.patch("src.ingestion.main.load_config")
    def test_main_runs_pipeline(self, mock_load_config, tmp_path, sample_df):
        """Positive: ingest.main() processes, validates, and saves data."""
        raw_dir = tmp_path / "raw"
        mock_load_config.return_value = PipelineConfig.model_validate(
            {
                "data": {
                    "raw_path": str(raw_dir) + "/",
                    "target_col": "price_eur_mwh",
                    "train_end": "2023-12-31",
                    "val_end": "2024-01-01",
                },
                "model": {"type": "sklearn.dummy.DummyRegressor"},
            }
        )

        with mock.patch(
            "src.ingestion.main.download_entsoe_data", return_value=sample_df
        ):
            ingest_mod.main()

        out = raw_dir / "entsoe_prices.csv"
        assert out.exists()
        loaded = pd.read_csv(out, parse_dates=["timestamp"])
        assert len(loaded) == len(sample_df)
        # Manifest is also written
        assert (raw_dir / "manifest.json").exists()


class TestFeaturiseMain:
    @mock.patch("src.features.main.load_config")
    def test_main_runs_pipeline(self, mock_load_config, tmp_path, sample_df):
        """Positive: featurise.main() featurises and materialises the superset.

        WS0 (scheme B): the featurise outputs are the superset parquet + its
        schema sidecar + manifest.json, plus the reference dataset. The old
        single ``features.parquet`` is deliberately NOT written.
        """
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(raw_dir / "entsoe_prices.csv", index=False)

        superset_dir = tmp_path / "processed" / "superset"
        reference = tmp_path / "reference" / "reference.parquet"

        mock_load_config.return_value = PipelineConfig.model_validate(
            {
                "data": {
                    "raw_path": str(raw_dir) + "/",
                    "superset_dir": str(superset_dir) + "/",
                    "reference_path": str(reference),
                    "target_col": "price_eur_mwh",
                    "train_end": "2023-12-31",
                    "val_end": "2024-01-01",
                },
                "features": {
                    "calendar": {"enabled": True},
                    "lags": {"enabled": True, "periods": [1, 2, 24]},
                    "derivatives": {
                        "enabled": False,
                        "order": [1, 2],
                        "smooth_windows": [3],
                    },
                },
                "model": {"type": "sklearn.dummy.DummyRegressor"},
            }
        )

        featurise_mod.main()

        parquet = superset_dir / "features_hourly.parquet"
        assert parquet.exists()
        assert (superset_dir / "features_hourly_schema.json").exists()
        assert (superset_dir / "manifest.json").exists()
        assert reference.exists()
        # Scheme B: the legacy single-file artifact must be gone.
        assert not (tmp_path / "processed" / "features.parquet").exists()

        features = pd.read_parquet(parquet)
        assert "hour" in features.columns
        assert "lag_1h" in features.columns


class TestTrainMain:
    @mock.patch("src.training.main.log_to_mlflow")
    @mock.patch("src.training.main.load_config")
    def test_main_trains_and_logs(
        self, mock_load_config, mock_log, tmp_path, sample_config
    ):
        """Positive: train.main() loads features, trains, and logs to MLflow."""
        from src.common.superset import build_manifest, superset_paths, write_manifest
        from src.features.main import (
            build_features,
            save_reference_data,
            train_val_test_split,
        )

        # 96 hours of history: the 24h lag warm-up NaNs eat the first day,
        # so the per-selection dropna still leaves a 24-row train split.
        raw = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2023-12-29 00:00:00", periods=96, freq="h", tz="UTC"
                ),
                "price_eur_mwh": np.linspace(40, 60, 96),
            }
        )

        superset_dir = tmp_path / "processed" / "superset"
        reference = tmp_path / "reference" / "reference.parquet"
        superset_dir.mkdir(parents=True, exist_ok=True)

        cfg = sample_config
        cfg.data.superset_dir = str(superset_dir) + "/"
        cfg.data.reference_path = str(reference)

        # WS0: featurise materialises the superset (parquet + schema +
        # manifest) plus the reference dataset. The legacy single parquet
        # is deliberately not written.
        frame, schema = build_features(
            raw, cfg.features, cfg.data.target_col, weather=None, drop_na=False
        )
        paths = superset_paths(cfg)
        frame.to_parquet(paths.parquet, index=False)
        schema.save(paths.schema)
        write_manifest(paths, build_manifest(paths, cfg, raw_csv_path=None))
        save_reference_data(
            train_val_test_split(frame.dropna(), cfg.data)[0], str(reference)
        )

        mock_load_config.return_value = cfg
        mock_log.return_value = "run-123"

        train_mod.main()

        assert paths.parquet.exists()
        assert paths.schema.exists()
        assert paths.manifest.exists()
        assert reference.exists()
        assert not (tmp_path / "processed" / "features.parquet").exists()

        features = pd.read_parquet(paths.parquet)
        assert "hour" in features.columns
        assert "lag_1h" in features.columns

        # Stage 2: three runs — XGBoost + persistence + seasonal naive
        assert mock_log.call_count == 3
        args, _ = mock_log.call_args
        assert "rmse" in args[1]
        run_names = [call[1].get("run_name") for call in mock_log.call_args_list]
        assert run_names == [
            "xgboost-v1",
            "baseline-persistence",
            "baseline-seasonal-naive",
        ]


class TestEvaluateMain:
    @mock.patch("src.evaluation.reporting.get_model_run_id", return_value=None)
    @mock.patch("src.evaluation.main.log_evaluation_results")
    @mock.patch("src.evaluation.main.load_model_from_registry")
    @mock.patch("src.evaluation.main.load_config")
    def test_main_evaluates(
        self,
        mock_load_config,
        mock_load_model,
        mock_log_results,
        mock_run_id,
        tmp_path,
        sample_config,
    ):
        """Positive: evaluate.main() loads test features and prints metrics."""
        from src.common.superset import build_manifest, superset_paths, write_manifest
        from src.features.main import build_features

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        sample_df_96 = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2023-12-29 00:00:00", periods=96, freq="h", tz="UTC"
                ),
                "price_eur_mwh": np.linspace(40, 60, 96),
            }
        )
        sample_df_96.to_csv(raw_dir / "entsoe_prices.csv", index=False)

        superset_dir = tmp_path / "processed" / "superset"
        superset_dir.mkdir(parents=True, exist_ok=True)

        cfg = sample_config
        cfg.data.raw_path = str(raw_dir) + "/"
        cfg.data.superset_dir = str(superset_dir) + "/"
        # Keep the test hermetic: plots would otherwise be written into the
        # project's reports/ directory.
        cfg.evaluation.generate_plots = False

        # WS0: featurise materialises the superset. A 96-row frame keeps a
        # non-empty 24-row test split after the warm-up NaN drop.
        frame, schema = build_features(
            sample_df_96, cfg.features, cfg.data.target_col, weather=None, drop_na=False
        )
        paths = superset_paths(cfg)
        frame.to_parquet(paths.parquet, index=False)
        schema.save(paths.schema)
        write_manifest(paths, build_manifest(paths, cfg, raw_csv_path=None))

        mock_load_config.return_value = cfg
        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.full(24, 50.0)
        mock_load_model.return_value = mock_model

        evaluate_mod.main()

        mock_model.predict.assert_called_once()
        mock_log_results.assert_called_once()
