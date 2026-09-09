"""
Unit tests for the main() CLI entry points of each pipeline module.

These exercise the orchestration functions by mocking load_config and
external services, so no real config or MLflow server is needed.
"""

from unittest import mock

import numpy as np
import pandas as pd

import src.evaluate as evaluate_mod
import src.featurise as featurise_mod
import src.ingest as ingest_mod
import src.train as train_mod


class TestIngestMain:
    @mock.patch("src.ingest.load_config")
    def test_main_runs_pipeline(self, mock_load_config, tmp_path, sample_df):
        """Positive: ingest.main() generates, validates, and saves data."""
        raw_dir = tmp_path / "raw"
        mock_load_config.return_value = {"data": {"raw_path": str(raw_dir) + "/"}}

        with mock.patch("src.ingest.generate_synthetic_data", return_value=sample_df):
            ingest_mod.main()

        out = raw_dir / "entsoe_prices.csv"
        assert out.exists()
        loaded = pd.read_csv(out, parse_dates=["timestamp"])
        assert len(loaded) == len(sample_df)
        # Manifest is also written
        assert (raw_dir / "manifest.json").exists()


class TestFeaturiseMain:
    @mock.patch("src.featurise.load_config")
    def test_main_runs_pipeline(self, mock_load_config, tmp_path, sample_df):
        """Positive: featurise.main() loads, featurises, splits, and saves."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(raw_dir / "entsoe_prices.csv", index=False)

        processed = tmp_path / "processed" / "features.parquet"
        reference = tmp_path / "reference" / "reference.parquet"

        mock_load_config.return_value = {
            "data": {
                "raw_path": str(raw_dir) + "/",
                "processed_path": str(processed),
                "reference_path": str(reference),
                "target_col": "price_eur_mwh",
                "train_end": "2023-12-31",
                "val_end": "2024-01-01",
            },
            "features": {
                "calendar": {"enabled": True},
                "lags": {"enabled": True, "periods": [1, 2, 24]},
                "derivatives": {"enabled": False, "order": [1, 2], "smooth_window": 3},
            },
        }

        featurise_mod.main()

        assert processed.exists()
        assert reference.exists()
        features = pd.read_parquet(processed)
        assert "hour" in features.columns
        assert "lag_1h" in features.columns


class TestTrainMain:
    @mock.patch("src.train.log_to_mlflow")
    @mock.patch("src.train.load_config")
    def test_main_trains_and_logs(
        self, mock_load_config, mock_log, tmp_path, sample_config
    ):
        """Positive: train.main() loads features, trains, and logs to MLflow."""
        processed = tmp_path / "features.parquet"
        sample_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2023-12-30", periods=72, freq="h", tz="UTC"
                ),
                "price_eur_mwh": np.linspace(40, 60, 72),
                "hour": list(range(24)) * 3,
            }
        )
        sample_df.to_parquet(processed, index=False)

        cfg = dict(sample_config)
        cfg["data"] = dict(sample_config["data"])
        cfg["data"]["processed_path"] = str(processed)
        mock_load_config.return_value = cfg
        mock_log.return_value = "run-123"

        train_mod.main()

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
    @mock.patch("src.evaluate.log_evaluation_results")
    @mock.patch("src.evaluate.load_model_from_registry")
    @mock.patch("src.evaluate.load_config")
    def test_main_evaluates(
        self, mock_load_config, mock_load_model, mock_log_results,
        tmp_path, sample_config
    ):
        """Positive: evaluate.main() loads test features and prints metrics."""
        processed = tmp_path / "features.parquet"
        sample_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2023-12-30", periods=72, freq="h", tz="UTC"
                ),
                "price_eur_mwh": np.linspace(40, 60, 72),
                "hour": list(range(24)) * 3,
            }
        )
        sample_df.to_parquet(processed, index=False)

        cfg = dict(sample_config)
        cfg["data"] = dict(sample_config["data"])
        cfg["data"]["processed_path"] = str(processed)
        mock_load_config.return_value = cfg

        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.full(24, 50.0)
        mock_load_model.return_value = mock_model

        evaluate_mod.main()

        mock_model.predict.assert_called_once()
        # Results are logged via log_evaluation_results (mocked — no real
        # MLflow HTTP calls, which would hang without a running server)
        mock_log_results.assert_called_once()
