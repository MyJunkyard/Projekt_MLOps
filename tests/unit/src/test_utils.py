"""
Unit tests for src/utils.py — load_config, compute_metrics, get_split_masks,
and setup_logging.
"""

import logging
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import yaml

from src.utils import (
    DEFAULT_LOG_FILE,
    compute_metrics,
    get_split_masks,
    load_config,
    setup_logging,
)


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

    def test_mape_with_negative_prices(self):
        """Positive: MAPE works with negative prices (masks zeros only)."""
        y_true = np.array([-10.0, 5.0, -20.0])
        y_pred = np.array([-12.0, 6.0, -22.0])
        result = compute_metrics(y_true, y_pred, ["mape"])
        # |(-10)-(-12)|/|-10| = 0.2, |5-6|/|5| = 0.2, |(-20)-(-22)|/|-20| = 0.1
        # Mean = (0.2 + 0.2 + 0.1)/3 * 100 = 16.666...
        assert result["mape"] == pytest.approx(16.666666666666664)

    def test_mape_mixed_signs(self):
        """Positive: MAPE handles mix of positive/negative/zero correctly."""
        y_true = np.array([0.0, 10.0, -5.0])
        y_pred = np.array([1.0, 12.0, -6.0])
        result = compute_metrics(y_true, y_pred, ["mape"])
        # Zero masked: |10-12|/10 = 20%, |-5-(-6)|/|-5| = 20% → mean = 20%
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


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
class TestSetupLogging:
    """Tests for setup_logging — isolated per-test via unique logger names.

    setup_logging adds at most one FileHandler and one StreamHandler per
    logger, keyed by handler type. Using a unique logger name per test keeps
    each case independent of the global ``src`` logger state.
    """

    def _flush(self, logger_name: str) -> None:
        """Flush all handlers attached to a (custom) logger so records hit disk."""
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            handler.flush()

    def test_creates_file_handler_and_writes_log(self, tmp_path):
        """Positive: setup_logging adds a FileHandler mirroring log records."""
        log_file = tmp_path / "logs" / "pipeline.log"
        name = f"src_ph_{uuid4().hex}"
        cfg = {"logging": {"level": "DEBUG", "file": str(log_file)}}

        logger = setup_logging(cfg, logger_name=name)
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)

        logging.getLogger(f"{name}.child").info("hello from file")
        self._flush(logger.name)

        assert log_file.exists()
        assert "hello from file" in log_file.read_text(encoding="utf-8")

    def test_default_file_in_tempdir(self):
        """Positive: with no logging.file, logs to DEFAULT_LOG_FILE in temp."""
        name = f"src_def_{uuid4().hex}"
        setup_logging({}, logger_name=name)

        logging.getLogger(f"{name}.child").info("default temp file check")
        self._flush(name)

        default_file = Path(DEFAULT_LOG_FILE)
        assert default_file.exists()
        assert "default temp file check" in default_file.read_text(encoding="utf-8")

    def test_custom_file_from_config(self, tmp_path):
        """Positive: cfg['logging.file'] overrides DEFAULT_LOG_FILE."""
        log_file = tmp_path / "custom" / "run.log"
        name = f"src_cus_{uuid4().hex}"
        setup_logging({"logging": {"file": str(log_file)}}, logger_name=name)

        logging.getLogger(f"{name}.child").warning("custom path check")
        self._flush(name)

        assert log_file.exists()
        assert "custom path check" in log_file.read_text(encoding="utf-8")
        assert str(log_file) != DEFAULT_LOG_FILE

    def test_idempotent_no_duplicate_file_handler(self, tmp_path):
        """Positive: repeated calls adjust level but do not add handlers."""
        log_file = tmp_path / "pipeline.log"
        name = f"src_idem_{uuid4().hex}"

        setup_logging(
            {"logging": {"level": "DEBUG", "file": str(log_file)}},
            logger_name=name,
        )
        setup_logging(
            {"logging": {"level": "INFO", "file": str(log_file)}},
            logger_name=name,
        )

        logger = logging.getLogger(name)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert logger.level == logging.INFO

    def test_missing_logging_section_falls_back_to_info(self):
        """Minimal config (no logging section) falls back to INFO + a handler."""
        name = f"src_min_{uuid4().hex}"
        logger = setup_logging({}, logger_name=name)
        assert logger.level == logging.INFO
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)