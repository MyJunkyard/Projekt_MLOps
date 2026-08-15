"""
Unit tests for src/utils.py — load_config, compute_metrics, get_split_masks.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.utils import compute_metrics, get_split_masks, load_config


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def test_load_config_returns_dict(self, tmp_path):
        """Positive: valid YAML returns a dict."""
        cfg_path = tmp_path / "params.yaml"
        cfg_path.write_text("data:\n  target_col: price_eur_mwh\n")
        result = load_config(str(cfg_path))
        assert result == {"data": {"target_col": "price_eur_mwh"}}

    def test_load_config_missing_file_raises(self, tmp_path):
        """Negative: missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "does_not_exist.yaml"))

    def test_load_config_invalid_yaml_raises(self, tmp_path):
        """Negative: malformed YAML raises yaml.YAMLError."""
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("data: [unclosed")
        with pytest.raises(yaml.YAMLError):
            load_config(str(cfg_path))


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------
class TestComputeMetrics:
    @pytest.mark.parametrize(
        "metric,expected",
        [
            ("rmse", 1.0),
            ("mae", 1.0),
            ("mape", 61.11111111111111),
            ("r2", -0.5),
        ],
    )
    def test_metric_values(self, metric, expected):
        """Positive: known arrays produce hand-computed metric values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        result = compute_metrics(y_true, y_pred, [metric])
        assert metric in result
        assert result[metric] == pytest.approx(expected)

    def test_perfect_prediction_zero_errors(self):
        """Positive: perfect predictions yield zero error metrics."""
        y_true = np.array([1.0, 2.0, 3.0])
        result = compute_metrics(y_true, y_true, ["rmse", "mae", "mape", "r2"])
        assert result["rmse"] == pytest.approx(0.0)
        assert result["mae"] == pytest.approx(0.0)
        assert result["mape"] == pytest.approx(0.0)
        assert result["r2"] == pytest.approx(1.0)

    def test_unknown_metric_ignored(self):
        """Negative: unknown metric keys are ignored."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        result = compute_metrics(y_true, y_pred, ["not_a_metric"])
        assert result == {}

    def test_mape_all_zero_true_is_nan(self):
        """Negative: MAPE with all-zero y_true returns NaN (no div-by-zero)."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = compute_metrics(y_true, y_pred, ["mape"])
        assert np.isnan(result["mape"])

    def test_mape_masks_zero_values(self):
        """Positive: MAPE ignores zero y_true entries."""
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([5.0, 12.0])
        result = compute_metrics(y_true, y_pred, ["mape"])
        # Only the non-zero entry contributes: |10-12|/10 * 100 = 20%
        assert result["mape"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# get_split_masks
# ---------------------------------------------------------------------------
class TestGetSplitMasks:
    def test_split_masks_partition_all_rows(self, sample_df, sample_config):
        """Positive: masks are mutually exclusive and cover all rows."""
        train, val, test = get_split_masks(sample_df, sample_config)
        assert train.sum() + val.sum() + test.sum() == len(sample_df)
        assert not (train & val).any()
        assert not (train & test).any()
        assert not (val & test).any()

    def test_split_masks_boundaries(self, sample_df, sample_config):
        """Positive: rows before train_end are train; val_end onward are test."""
        train, val, test = get_split_masks(sample_df, sample_config)

        # sample_df starts 2023-12-30, train_end 2023-12-31
        assert bool(train[0])
        # 24 rows before 2023-12-31 are train
        assert train.sum() == 24
        # 24 rows between 2023-12-31 and 2024-01-01 are val
        assert val.sum() == 24
        # 24 rows on/after 2024-01-01 are test
        assert test.sum() == 24

    def test_split_masks_returns_boolean_arrays(self, sample_df, sample_config):
        """Positive: returned masks are boolean numpy arrays."""
        train, val, test = get_split_masks(sample_df, sample_config)
        assert train.dtype == bool
        assert val.dtype == bool
        assert test.dtype == bool