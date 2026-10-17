"""
Unit tests for the main() CLI entry points of each pipeline module.

These exercise the orchestration functions by mocking load_config and
external services, so no real config or MLflow server is needed.
"""

from unittest import mock

import numpy as np
import pandas as pd
import pytest

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
    @staticmethod
    def _materialise_superset(tmp_path, cfg):
        """Build the WS0 superset + reference that ``train.main()`` reads.

        96 hours of history: the 24h lag warm-up NaNs eat the first day, so
        the per-selection dropna leaves a 24-row train/val/test split each.
        Mutates ``cfg.data`` in place (superset_dir + reference_path under
        ``tmp_path``) and returns ``(paths, reference)``.
        """
        from src.common.superset import (
            build_manifest,
            superset_paths,
            write_manifest,
        )
        from src.features.main import (
            build_features,
            save_reference_data,
            train_val_test_split,
        )

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
        return paths, reference

    @mock.patch("src.training.main.log_to_mlflow")
    @mock.patch("src.training.main.load_config")
    def test_main_trains_and_logs(
        self, mock_load_config, mock_log, tmp_path, sample_config
    ):
        """Positive: train.main() loads features, trains, and logs to MLflow."""
        cfg = sample_config
        paths, reference = self._materialise_superset(tmp_path, cfg)
        # WS4: pin the val-split analysis OFF — this test pins the
        # train-and-log flow only; the flag defaults to true in config.
        cfg.evaluation.run_analysis_after_train = False

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

    @mock.patch("src.training.main.log_evaluation_results")
    @mock.patch("src.training.main.run_analysis")
    @mock.patch("src.training.main.log_to_mlflow")
    @mock.patch("src.training.main.load_config")
    def test_main_runs_val_analysis_when_flag_true(
        self,
        mock_load_config,
        mock_log,
        mock_run_analysis,
        mock_log_eval_results,
        tmp_path,
        sample_config,
    ):
        """WS4: flag true → val split analysed, artifacts on the primary run."""
        from src.evaluation.analysis import AnalysisResult

        cfg = sample_config
        self._materialise_superset(tmp_path, cfg)
        cfg.evaluation.run_analysis_after_train = True
        cfg.evaluation.generate_plots = False  # hermetic (mocked anyway)
        mock_load_config.return_value = cfg
        mock_log.return_value = "run-123"
        canned = AnalysisResult(
            metrics={"rmse": 1.0},
            artifact_paths=("val_residual_breakdown.csv",),
        )
        mock_run_analysis.return_value = canned

        train_mod.main()

        mock_run_analysis.assert_called_once()
        args, kwargs = mock_run_analysis.call_args
        # The fitted primary model, the val split of the SAME clean frame
        # train just fitted on, and the model's own feature columns.
        assert args[1] is mock_log.call_args_list[0][0][0]
        assert isinstance(args[2], pd.DataFrame) and len(args[2]) == 24
        assert args[3] == "val"  # split prefix drives the artifact names
        assert args[4] == mock_log.call_args_list[0][1]["feature_names"]
        assert kwargs == {}  # output_dir defaults to "reports"
        # Artifacts-only attach (empty metrics) to the PRIMARY run id.
        mock_log_eval_results.assert_called_once_with(
            cfg, {}, list(canned.artifact_paths), run_id="run-123"
        )

    @mock.patch("src.training.main.run_analysis")
    @mock.patch("src.training.main.log_to_mlflow")
    @mock.patch("src.training.main.load_config")
    def test_main_skips_val_analysis_when_flag_false(
        self,
        mock_load_config,
        mock_log,
        mock_run_analysis,
        tmp_path,
        sample_config,
    ):
        """WS4: flag false → training completes with no analysis call."""
        cfg = sample_config
        self._materialise_superset(tmp_path, cfg)
        cfg.evaluation.run_analysis_after_train = False
        mock_load_config.return_value = cfg
        mock_log.return_value = "run-123"

        train_mod.main()

        mock_run_analysis.assert_not_called()

    @mock.patch(
        "src.training.main.log_evaluation_results",
        side_effect=Exception("mlflow down"),
    )
    @mock.patch("src.training.main.run_analysis")
    @mock.patch("src.training.main.log_to_mlflow")
    @mock.patch("src.training.main.load_config")
    def test_main_survives_artifact_attach_failure(
        self,
        mock_load_config,
        mock_log,
        mock_run_analysis,
        mock_log_eval_results,
        tmp_path,
        sample_config,
    ):
        """WS4: a failing artifact attach is a WARNING, never a failed run."""
        from src.evaluation.analysis import AnalysisResult

        cfg = sample_config
        self._materialise_superset(tmp_path, cfg)
        cfg.evaluation.run_analysis_after_train = True
        cfg.evaluation.generate_plots = False
        mock_load_config.return_value = cfg
        mock_log.return_value = "run-123"
        mock_run_analysis.return_value = AnalysisResult(
            metrics={"rmse": 1.0},
            artifact_paths=("val_residual_breakdown.csv",),
        )

        train_mod.main()  # must not raise

        mock_run_analysis.assert_called_once()
        mock_log_eval_results.assert_called_once()


class TestEvaluateMain:
    # Two get_model_run_id patches, both returning None: main() resolves the
    # scored run once through its OWN imported name (src.evaluation.main.
    # get_model_run_id), and resolve_evaluate_columns — which runs for real
    # here — resolves internally through src.evaluation.reporting.
    # get_model_run_id. Patching both keeps the test hermetic (no live
    # MlflowClient HTTP call) and treats the champion as unresolvable, so
    # columns fall back to features.select.
    @mock.patch("src.evaluation.main.get_model_run_id", return_value=None)
    @mock.patch("src.evaluation.reporting.get_model_run_id", return_value=None)
    @mock.patch("src.evaluation.main.log_evaluation_results")
    @mock.patch("src.evaluation.main.load_model_from_registry")
    @mock.patch("src.evaluation.main.load_config")
    def test_main_evaluates(
        self,
        mock_load_config,
        mock_load_model,
        mock_log_results,
        mock_reporting_run_id,
        mock_main_run_id,
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

    @mock.patch("src.evaluation.main.get_model_run_id", return_value=None)
    @mock.patch("src.evaluation.reporting.get_model_run_id", return_value=None)
    @mock.patch("src.evaluation.main.log_evaluation_results")
    @mock.patch("src.evaluation.main.load_model_from_registry")
    @mock.patch("src.evaluation.main.load_config")
    def test_main_delegates_to_run_analysis(
        self,
        mock_load_config,
        mock_load_model,
        mock_log_results,
        mock_reporting_run_id,
        mock_main_run_id,
        superset_dir,
        sample_config,
    ):
        """WS4: the test split is delegated to ``run_analysis`` (D4-b/D4-c).

        The canned result's metrics and split-prefixed artifact paths must
        reach ``log_evaluation_results`` untouched — this pins the
        delegation seam between ``main`` and the orchestrator.
        """
        from pathlib import Path

        from src.evaluation.analysis import AnalysisResult

        cfg = sample_config  # superset_dir already repointed at tmp_path
        # Hermetic: no reports/ writes (run_analysis is mocked anyway).
        cfg.evaluation.generate_plots = False
        mock_load_config.return_value = cfg
        mock_model = mock.MagicMock()
        mock_load_model.return_value = mock_model

        canned = AnalysisResult(
            metrics={"rmse": 1.0, "mae": 0.5},
            artifact_paths=("test_residual_breakdown.csv",),
        )
        with mock.patch(
            "src.evaluation.main.run_analysis", return_value=canned
        ) as mock_run:
            evaluate_mod.main()

        args, kwargs = mock_run.call_args
        assert args[1] is mock_model  # the registry-loaded model
        assert isinstance(args[2], pd.DataFrame) and len(args[2]) > 0
        assert args[3] == "test"  # split prefix drives the artifact names
        assert isinstance(args[4], list) and args[4]  # resolved feature cols
        assert kwargs == {"output_dir": Path("reports")}
        mock_log_results.assert_called_once_with(
            cfg,
            canned.metrics,
            list(canned.artifact_paths),
            run_id=None,
        )


class TestCliDispatch:
    """``python -m src`` dispatch: evaluate --run-id and report (WS6 D10/D12).

    MLflow is mocked throughout — no tracking server in tier 1 (plan §8).
    """

    def test_evaluate_run_id_reaches_evaluation_main(self):
        """Positive: `evaluate --run-id X` calls evaluation.main(run_id='X')."""
        import src.cli as cli

        with mock.patch.object(cli, "evaluate_main") as mock_eval:
            with mock.patch("sys.argv", ["src", "evaluate", "--run-id", "abc123"]):
                cli.main()
        mock_eval.assert_called_once_with(run_id="abc123")

    def test_evaluate_default_run_id_none(self):
        """Positive: bare `evaluate` passes run_id=None (champion alias)."""
        import src.cli as cli

        with mock.patch.object(cli, "evaluate_main") as mock_eval:
            with mock.patch("sys.argv", ["src", "evaluate"]):
                cli.main()
        mock_eval.assert_called_once_with(run_id=None)

    def test_report_dispatches_to_report_main(self):
        """Positive: `report --series B` reaches report.main with the flag."""
        import src.cli as cli

        with mock.patch.object(cli, "report_main") as mock_report:
            with mock.patch("sys.argv", ["src", "report", "--series", "B"]):
                cli.main()
        mock_report.assert_called_once()
        argv = mock_report.call_args.kwargs["argv"]
        assert "--series" in argv and "B" in argv

    def test_report_dm_errors_naming_ws7(self):
        """Negative: `report --dm` builds a call whose args trigger the WS7 error."""
        import src.cli as cli
        from src.evaluation import report as report_mod

        captured = {}

        def _capture(argv):
            captured["argv"] = argv

        with mock.patch.object(cli, "report_main", side_effect=_capture):
            with mock.patch("sys.argv", ["src", "report", "--dm"]):
                cli.main()
        # The dispatch forwards --dm; the report parser then rejects it naming WS7.
        assert "--dm" in captured["argv"]
        with mock.patch("src.config.load_config"):
            with pytest.raises(NotImplementedError, match="WS7"):
                report_mod.main(captured["argv"])

