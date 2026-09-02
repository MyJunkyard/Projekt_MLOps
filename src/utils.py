"""
utils.py — Shared helpers for pipeline modules.

Common utilities used across the pipeline (config loading, logging setup,
metric computation, train/val/test splitting) live here to avoid
duplication and keep a single source of truth.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Parent logger of all pipeline modules ("src.ingest", "src.train", ...).
PACKAGE_LOGGER_NAME = "src"

# Default log file: a temp subfolder inside the project (project_root/logs),
# so logs are easy to find and read when debugging a pipeline run. Override
# via ``cfg["logging"]["file"]``. ``PROJECT_ROOT`` is resolved relative to
# this file so the path is stable regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILE = str(DEFAULT_LOG_DIR / "energy_forecast_pipeline.log")


def setup_logging(cfg: dict, logger_name: str = PACKAGE_LOGGER_NAME) -> logging.Logger:
    """Configure logging for the pipeline package.

    Attaches a console handler (stderr) **and** a file handler to the
    ``"src"`` package logger (not the root logger, so third-party
    library logging is untouched) with the shared ``LOG_FORMAT`` and the
    level from ``cfg["logging"]["level"]``. All modules log via
    ``logging.getLogger(__name__)``, which resolves to a child of this
    logger and therefore inherits the handlers and level.

    The file handler writes a copy of all pipeline logs to
    ``cfg["logging"]["file"]`` if set, otherwise to
    ``<project_root>/logs/energy_forecast_pipeline.log``
    (``utils.DEFAULT_LOG_FILE``), so runs remain traceable even when stderr
    is lost (e.g. via Make) and are easy to find in the project.

    Safe to call multiple times (repeat calls only adjust the level).
    The level falls back to ``INFO`` when the ``logging`` section is
    missing (e.g. in tests with minimal configs). Records still
    propagate to the root logger (which has no handler by default), so
    pytest's ``caplog`` fixture keeps working. Logging state is
    per-process: code that spawns worker processes must call this
    function again in each child. See ``docs/logging.md`` for the full
    logging policy.

    Args:
        cfg: Configuration dictionary (from ``load_config``) with an
            optional ``logging.level`` key (DEBUG | INFO | WARNING | ERROR)
            and an optional ``logging.file`` key (path to the log file).
        logger_name: Logger to configure. Defaults to the package logger.

    Returns:
        The configured logger.
    """
    level_name = str(cfg.get("logging", {}).get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

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
    # (override with cfg["logging"]["file"]).
    log_file = str(cfg.get("logging", {}).get("file", DEFAULT_LOG_FILE))
    if not any(isinstance(h, logging.FileHandler) for h in package_logger.handlers):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        package_logger.addHandler(file_handler)

    package_logger.setLevel(level)
    return package_logger


def load_config(config_path: str = "params.yaml") -> dict:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary of configuration values.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    metrics_list: list[str],
) -> dict[str, float]:
    """Compute requested regression metrics and return as a dict.

    Args:
        y_true: Ground-truth target values. Array-like of shape
            (n_samples,); normalized to a 1-D float ndarray via
            ``np.asarray``.
        y_pred: Predicted target values. Array-like of shape
            (n_samples,); normalized the same way.
        metrics_list: Names of metrics to compute. Supported values:
            ``"rmse"``, ``"mae"``, ``"mape"``, ``"r2"``.

    Returns:
        A dict mapping each requested metric name to its float value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    results = {}
    for metric in metrics_list:
        if metric == "rmse":
            results["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif metric == "mae":
            results["mae"] = float(mean_absolute_error(y_true, y_pred))
        elif metric == "mape":
            # Avoid division by zero — mask zero prices
            mask = y_true != 0
            if mask.sum() > 0:
                results["mape"] = float(
                    np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                )
            else:
                results["mape"] = float("nan")
        elif metric == "r2":
            results["r2"] = float(r2_score(y_true, y_pred))
    return results


def get_split_masks(
    df: pd.DataFrame, cfg: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for train/val/test splits based on config dates.

    Required DataFrame contract:
        - ``timestamp``: timezone-aware datetime column (UTC), one row
          per period, no duplicate timestamps.

    Args:
        df: DataFrame meeting the contract above.
        cfg: Configuration dict with ``data.train_end`` and ``data.val_end``
            boundaries (tz-naive date strings, interpreted as UTC).

    Returns:
        A tuple ``(train_mask, val_mask, test_mask)`` of boolean numpy
        arrays, each of shape (n_samples,), mutually exclusive and
        collectively covering all rows of ``df``.
    """
    # Boundaries are tz-naive in params.yaml; data is tz-aware (UTC)
    train_end = pd.Timestamp(cfg["data"]["train_end"], tz="UTC")
    val_end = pd.Timestamp(cfg["data"]["val_end"], tz="UTC")

    train_mask = (df["timestamp"] < train_end).to_numpy()
    val_mask = ((df["timestamp"] >= train_end) & (df["timestamp"] < val_end)).to_numpy()
    test_mask = (df["timestamp"] >= val_end).to_numpy()
    return train_mask, val_mask, test_mask