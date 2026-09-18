"""
common/logsetup.py — Logging setup shared by all pipeline modules.

Moved verbatim from ``utils.py`` (Workstream 0 module restructure): the
role-separation argument applied to our own helpers — logging belongs in
its own module. Named ``logsetup`` (not ``logging``) to avoid shadowing
the stdlib module.

Takes the ``LoggingConfig`` sub-model (the narrowest config it
consumes).
"""

import logging
import sys
from pathlib import Path

from src.config.models import LoggingConfig

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Parent logger of all pipeline modules ("src.ingestion.main",
# "src.training.main", ...).
PACKAGE_LOGGER_NAME = "src"

# Default log file: a temp subfolder inside the project (project_root/logs),
# so logs are easy to find and read when debugging a pipeline run. Override
# via ``logging.file`` in params.yaml. ``PROJECT_ROOT`` is resolved relative
# to this file so the path is stable regardless of the current working
# directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILE = str(DEFAULT_LOG_DIR / "energy_forecast_pipeline.log")


def setup_logging(
    logging_cfg: LoggingConfig, logger_name: str = PACKAGE_LOGGER_NAME
) -> logging.Logger:
    """Configure logging for the pipeline package.

    Attaches a console handler (stderr) **and** a file handler to the
    ``"src"`` package logger (not the root logger, so third-party
    library logging is untouched) with the shared ``LOG_FORMAT`` and the
    level from ``logging_cfg.level``. All modules log via
    ``logging.getLogger(__name__)``, which resolves to a child of this
    logger and therefore inherits the handlers and level.

    The file handler writes a copy of all pipeline logs to
    ``logging_cfg.file`` if set, otherwise to
    ``<project_root>/logs/energy_forecast_pipeline.log``
    (``logsetup.DEFAULT_LOG_FILE``), so runs remain traceable even when stderr
    is lost (e.g. via Make) and are easy to find in the project.

    Safe to call multiple times (repeat calls only adjust the level).
    Records still propagate to the root logger (which has no handler by
    default), so pytest's ``caplog`` fixture keeps working. Logging state
    is per-process: code that spawns worker processes must call this
    function again in each child. See ``docs/logging.md`` for the full
    logging policy.

    Args:
        logging_cfg: ``LoggingConfig`` with the ``level``
            (DEBUG | INFO | WARNING | ERROR) and an optional ``file``
            path override.
        logger_name: Logger to configure. Defaults to the package logger.

    Returns:
        The configured logger.
    """
    level = getattr(logging, logging_cfg.level, logging.INFO)

    package_logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler (stderr) — added once per process.
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in package_logger.handlers
    ):
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        package_logger.addHandler(console_handler)

    # File handler — mirror logs to <project_root>/logs/energy_forecast_pipeline.log
    # (override with logging_cfg.file).
    log_file = logging_cfg.file or DEFAULT_LOG_FILE
    if not any(isinstance(h, logging.FileHandler) for h in package_logger.handlers):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        package_logger.addHandler(file_handler)

    package_logger.setLevel(level)
    return package_logger
