"""
Unit tests for the common package — setup_logging and get_split_masks.

Config loading/validation tests live in ``tests/unit/config/``;
``compute_metrics`` has its own module since WS7 in
``tests/unit/common/test_metrics.py``.
"""

import logging
from pathlib import Path
from uuid import uuid4

from src.common.logsetup import DEFAULT_LOG_FILE, setup_logging
from src.common.splits import get_split_masks
from src.config.models import LoggingConfig


# ---------------------------------------------------------------------------
# get_split_masks
# ---------------------------------------------------------------------------
class TestGetSplitMasks:
    def test_split_masks_partition_all_rows(self, sample_df, sample_config):
        """Positive: masks are mutually exclusive and cover all rows."""
        train, val, test = get_split_masks(sample_df, sample_config.data)
        assert train.sum() + val.sum() + test.sum() == len(sample_df)
        assert not (train & val).any()
        assert not (train & test).any()
        assert not (val & test).any()

    def test_split_masks_boundaries(self, sample_df, sample_config):
        """Positive: rows before train_end are train; val_end onward are test."""
        train, val, test = get_split_masks(sample_df, sample_config.data)

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
        train, val, test = get_split_masks(sample_df, sample_config.data)
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
        logging_cfg = LoggingConfig(level="DEBUG", file=str(log_file))

        logger = setup_logging(logging_cfg, logger_name=name)
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)

        logging.getLogger(f"{name}.child").info("hello from file")
        self._flush(logger.name)

        assert log_file.exists()
        assert "hello from file" in log_file.read_text(encoding="utf-8")

    def test_default_file_in_tempdir(self):
        """Positive: with no logging.file, logs to DEFAULT_LOG_FILE in temp."""
        name = f"src_def_{uuid4().hex}"
        setup_logging(LoggingConfig(), logger_name=name)

        logging.getLogger(f"{name}.child").info("default temp file check")
        self._flush(name)

        default_file = Path(DEFAULT_LOG_FILE)
        assert default_file.exists()
        assert "default temp file check" in default_file.read_text(encoding="utf-8")

    def test_custom_file_from_config(self, tmp_path):
        """Positive: logging.file overrides DEFAULT_LOG_FILE."""
        log_file = tmp_path / "custom" / "run.log"
        name = f"src_cus_{uuid4().hex}"
        setup_logging(LoggingConfig(file=str(log_file)), logger_name=name)

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
            LoggingConfig(level="DEBUG", file=str(log_file)),
            logger_name=name,
        )
        setup_logging(
            LoggingConfig(level="INFO", file=str(log_file)),
            logger_name=name,
        )

        logger = logging.getLogger(name)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert logger.level == logging.INFO

    def test_default_config_level_is_info(self):
        """Default LoggingConfig resolves to INFO + a file handler."""
        name = f"src_min_{uuid4().hex}"
        logger = setup_logging(LoggingConfig(), logger_name=name)
        assert logger.level == logging.INFO
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    def test_level_case_insensitive(self, tmp_path):
        """Positive: lowercase level names work (normalized by the model)."""
        name = f"src_ci_{uuid4().hex}"
        logging_cfg = LoggingConfig.model_validate(
            {"level": "debug", "file": str(tmp_path / "ci.log")}
        )
        logger = setup_logging(logging_cfg, logger_name=name)
        assert logger.level == logging.DEBUG
