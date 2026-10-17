"""
Unit tests for the WS4 residual-analysis orchestrator
(``src/evaluation/analysis.py``).

Covers the import guard (D4-a/D4-h: the evaluation package must pull in no
``src.training``), the split-prefixed artifact contract (D4-c), the
``generate_plots: false`` hermiticity contract, and the single-owner rule for
the sample-week window (D4-d).
"""

import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.common.splits import get_split_masks
from src.evaluation.analysis import AnalysisResult, run_analysis

REPO_ROOT = Path(__file__).resolve().parents[3]


class _ConstantModel:
    """Stub exposing the ``.predict(X)`` surface ``run_analysis`` needs."""

    def __init__(self, value: float = 50.0) -> None:
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def _split_frame(df, cfg, split: str):
    """Row-slice ``df`` to one split, mirroring the main() call-sites."""
    train_mask, val_mask, test_mask = get_split_masks(df, cfg.data)
    mask = {"train": train_mask, "val": val_mask, "test": test_mask}[split]
    return df[mask].reset_index(drop=True)


class TestEvaluationImportGuard:
    def test_evaluation_package_imports_no_src_training(self):
        """Regression guard (D4-a/D4-h): no ``evaluation -> training`` edge.

        Runs in a subprocess so a ``src.training`` module imported by an
        earlier test in this process cannot mask a real violation.
        """
        code = (
            "import sys\n"
            "import src.evaluation.analysis\n"
            "import src.evaluation.main\n"
            "import src.evaluation.reporting\n"
            "bad = sorted(m for m in sys.modules"
            " if m.startswith('src.training'))\n"
            "assert not bad, f'evaluation pulled in src.training: {bad}'\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=180,
        )
        assert proc.returncode == 0, (
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )


class TestAnalysisResult:
    def test_is_frozen(self):
        """D4-b: the result is an immutable frozen dataclass."""
        import dataclasses

        result = AnalysisResult(metrics={"rmse": 1.0}, artifact_paths=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.metrics = {}


class TestRunAnalysis:
    def test_writes_split_prefixed_artifacts(
        self, tmp_path, sample_config, sample_df_with_features
    ):
        """Positive: full artifact set, split-prefixed, all files non-empty."""
        df = _split_frame(sample_df_with_features, sample_config, "test")
        out = tmp_path / "reports"

        result = run_analysis(
            sample_config, _ConstantModel(), df, "test", ["hour"],
            output_dir=out,
        )

        assert isinstance(result, AnalysisResult)
        names = sorted(Path(p).name for p in result.artifact_paths)
        assert names == [
            "test_actual_vs_predicted.png",
            "test_residual_breakdown.csv",
            "test_residual_breakdown.png",
            "test_residuals_over_time.png",
        ]
        for artifact in result.artifact_paths:
            path = Path(artifact)
            assert path.exists()
            assert path.stat().st_size > 0
        # Metrics still flow even with plots on (single source of truth).
        assert sorted(result.metrics) == ["mae", "rmse"]
        assert all(np.isfinite(v) for v in result.metrics.values())

    def test_generate_plots_false_writes_nothing(
        self, tmp_path, sample_config, sample_df_with_features
    ):
        """Negative: plots off -> metrics returned, no ``reports/`` writes."""
        sample_config.evaluation.generate_plots = False
        df = _split_frame(sample_df_with_features, sample_config, "test")
        out = tmp_path / "reports"

        result = run_analysis(
            sample_config, _ConstantModel(), df, "test", ["hour"],
            output_dir=out,
        )

        assert result.artifact_paths == ()
        assert sorted(result.metrics) == ["mae", "rmse"]
        assert not out.exists()

    def test_sample_week_inside_split_emits_window(
        self, tmp_path, sample_config, sample_df_with_features
    ):
        """Positive (D4-d): ``sample_week`` in this split -> window artifact."""
        sample_config.evaluation.sample_week = date(2024, 1, 1)
        df = _split_frame(sample_df_with_features, sample_config, "test")
        out = tmp_path / "reports"

        result = run_analysis(
            sample_config, _ConstantModel(), df, "test", ["hour"],
            output_dir=out,
        )

        names = {Path(p).name for p in result.artifact_paths}
        assert "test_actual_vs_predicted_window.png" in names
        assert len(result.artifact_paths) == 5

    def test_sample_week_outside_split_warns_and_skips(
        self, tmp_path, caplog, sample_config, sample_df_with_features
    ):
        """Negative (D4-d): window outside the split -> WARNING, no artifact.

        The val split ends before ``sample_week`` starts, so only the test
        split may own the window; val must warn and emit none.
        """
        sample_config.evaluation.sample_week = date(2024, 1, 1)
        df = _split_frame(sample_df_with_features, sample_config, "val")
        out = tmp_path / "reports"

        with caplog.at_level(logging.WARNING, logger="src.evaluation.analysis"):
            result = run_analysis(
                sample_config, _ConstantModel(), df, "val", ["hour"],
                output_dir=out,
            )

        names = {Path(p).name for p in result.artifact_paths}
        assert "val_actual_vs_predicted_window.png" not in names
        assert len(result.artifact_paths) == 4
        assert "not inside the val split" in caplog.text
