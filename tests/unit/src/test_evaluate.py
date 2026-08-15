"""
Unit tests for src/evaluate.py — test feature loading and results table.
"""

import pandas as pd
import pytest

from src.evaluate import load_test_features, print_results_table


class TestLoadTestFeatures:
    def test_returns_x_and_y(self, features_parquet_path, sample_config):
        """Positive: returns X_test and y_test arrays."""
        X_test, y_test = load_test_features(features_parquet_path, sample_config)
        assert X_test.shape[0] == y_test.shape[0]
        assert X_test.shape[0] > 0

    def test_missing_file_raises(self, tmp_path, sample_config):
        """Negative: missing parquet raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_test_features(str(tmp_path / "nope.parquet"), sample_config)


class TestPrintResultsTable:
    def test_prints_metrics(self, capsys):
        """Positive: prints formatted metrics table."""
        print_results_table({"rmse": 1.2345, "mae": 0.9876})
        captured = capsys.readouterr()
        assert "Test Set Evaluation Results" in captured.out
        assert "RMSE" in captured.out
        assert "MAE" in captured.out
        assert "1.2345" in captured.out
        assert "0.9876" in captured.out