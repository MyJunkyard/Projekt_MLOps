"""
train.py — Stage 2: Model training with XGBoost, baselines, and MLflow logging.

Loads features from Parquet, dynamically instantiates the model class
specified in params.yaml, trains on the train split, evaluates on val,
and logs everything to MLflow. Also trains persistence and seasonal-naive
baselines as separate MLflow runs for comparison.
"""

import hashlib
import importlib
import json
import logging
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from mlflow.sklearn import log_model

from src.utils import compute_metrics, get_split_masks, load_config, setup_logging

# Stable module logger name (not __name__ — `python -m` sets __name__ to
# "__main__", which would bypass the configured src handlers).
MODULE_LOGGER_NAME = "src.train"
logger = logging.getLogger(MODULE_LOGGER_NAME)


class PersistenceModel:
    """Persistence baseline: predict = last observed value (lag-1).

    A simple model that stores the last training target value and predicts
    it for all future inputs. This is the standard persistence baseline
    for time-series forecasting.
    """

    def __init__(self):
        self.last_value: float = 0.0

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> "PersistenceModel":
        """Store the last observed target value.

        Args:
            X: Array-like of shape (n_samples, n_features). Unused —
                persistence uses no features; accepted for API symmetry.
            y: 1-D array of target values in chronological order.
                Normalized via ``np.asarray``.
        """
        y = np.asarray(y)
        self.last_value = float(y[-1])
        return self

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """Predict the last observed value for all inputs.

        Args:
            X: Array-like of shape (n_samples, n_features). Only the
                number of rows is used; feature values are ignored.
                Normalized via ``np.asarray`` (ndarray, DataFrame, or
                nested sequences are all accepted).

        Returns:
            1-D float array of shape (n_samples,) filled with the last
            observed target value.
        """
        n = np.asarray(X).shape[0]
        return np.full(n, self.last_value)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return uniform importances (persistence uses no features)."""
        return np.array([])


class SeasonalNaiveModel:
    """Seasonal naive baseline: predict = same hour last week (lag-168).

    A simple model that stores the last 168 training target values and
    predicts them cyclically. This captures weekly seasonality.
    """

    def __init__(self, season_length: int = 168):
        self.season_length = season_length
        self.history: np.ndarray = np.array([])

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> "SeasonalNaiveModel":
        """Store the last ``season_length`` target values.

        Args:
            X: Array-like of shape (n_samples, n_features). Unused —
                seasonal naive uses no features; accepted for API symmetry.
            y: 1-D array of target values in chronological order.
                Normalized via ``np.asarray``.
        """
        y = np.asarray(y)
        self.history = y[-self.season_length:]
        return self

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """Predict by cycling through the stored seasonal history.

        Args:
            X: Array-like of shape (n_samples, n_features). Only the
                number of rows is used; feature values are ignored.
                Normalized via ``np.asarray`` (ndarray, DataFrame, or
                nested sequences are all accepted).

        Returns:
            1-D float array of shape (n_samples,) cycling through the
            stored seasonal history (zeros when the history is empty).
        """
        n = np.asarray(X).shape[0]
        if len(self.history) == 0:
            return np.zeros(n)
        indices = np.arange(n) % len(self.history)
        return self.history[indices]

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return uniform importances (seasonal naive uses no features)."""
        return np.array([])


def load_model(cfg: dict):
    """Dynamically load a model class from its fully qualified name.

    Example: "sklearn.dummy.DummyRegressor" → sklearn.dummy.DummyRegressor

    Args:
        cfg: Configuration dict with ``model.type`` (dotted path) and
            ``model.params`` (constructor kwargs).

    Returns:
        An instantiated model object.

    Raises:
        ImportError: If the module path cannot be imported.
        AttributeError: If the class name is not found in the module.
    """
    module_path, class_name = cfg["model"]["type"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(**cfg["model"]["params"])


def load_features(path: str, cfg: dict) -> tuple:
    """Load features.parquet and separate features (X) from target (y).

    Args:
        path: Path to the features Parquet file.
        cfg: Configuration dict with ``data.target_col`` and split dates.

    Returns:
        A tuple ``(X_train, y_train, X_val, y_val, X_test, y_test)`` of
        numpy arrays.
    """
    df = pd.read_parquet(path)

    # Separate target
    target_col = cfg["data"]["target_col"]
    y = df[target_col].values

    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c not in [target_col, "timestamp"]]
    X = df[feature_cols].values

    # Reconstruct splits from the concatenated data using shared split logic
    train_mask, val_mask, test_mask = get_split_masks(df, cfg)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info("Features: %d columns", len(feature_cols))
    logger.debug("X_train: %s, y_train: %s", X_train.shape, y_train.shape)
    logger.debug("X_val: %s, y_val: %s", X_val.shape, y_val.shape)
    logger.debug("X_test: %s, y_test: %s", X_test.shape, y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_feature_names(path: str, cfg: dict) -> list[str]:
    """Get the list of feature column names from the features parquet.

    Args:
        path: Path to the features Parquet file.
        cfg: Configuration dict with ``data.target_col``.

    Returns:
        A list of feature column names (excluding target and timestamp).
    """
    df = pd.read_parquet(path)
    target_col = cfg["data"]["target_col"]
    feature_cols = [c for c in df.columns if c not in [target_col, "timestamp"]]
    return feature_cols


def get_git_commit_hash() -> str:
    """Get the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def compute_params_hash(config_path: str = "params.yaml") -> str:
    """Compute SHA256 hash of the params.yaml file.

    The hash serves as a quick fingerprint for comparing runs — two runs
    with the same hash used identical configuration. For full reproducibility,
    the params.yaml file itself is also logged as an MLflow artifact (see
    ``log_to_mlflow``).

    Args:
        config_path: Path to the params.yaml file.

    Returns:
        The SHA256 hex digest of the file contents, or "unknown" if the
        file does not exist.
    """
    path_obj = Path(config_path)
    if not path_obj.exists():
        return "unknown"
    return hashlib.sha256(path_obj.read_bytes()).hexdigest()


def log_feature_importances(model, feature_names: list[str]) -> None:
    """Log model feature importances as a JSON artifact to MLflow.

    Args:
        model: A fitted model with a ``feature_importances_`` attribute.
        feature_names: List of feature column names.
    """
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    if len(importances) == 0:
        return

    importance_dict = {
        name: float(imp)
        for name, imp in zip(feature_names, importances, strict=False)
    }

    # Sort by importance descending
    importance_dict = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    # Log as JSON artifact
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(importance_dict, f, indent=2)
        temp_path = f.name

    mlflow.log_artifact(temp_path, artifact_path="feature_importances")
    Path(temp_path).unlink()


def train_baseline_persistence(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple:
    """Train a persistence baseline model.

    Persistence baseline: predict tomorrow = today (lag-1 model).
    The prediction for each row is the last observed value.

    Args:
        X_train: Training features (unused — persistence doesn't use features).
        y_train: Training target values.
        X_val: Validation features (unused).
        y_val: Validation target values.

    Returns:
        A tuple ``(model, metrics)`` where model is a fitted
        ``PersistenceModel`` and metrics is a dict.
    """
    model = PersistenceModel()
    model.fit(X_train, y_train)
    y_pred = np.asarray(model.predict(X_val))
    metrics = compute_metrics(y_val, y_pred, ["rmse", "mae"])
    return model, metrics


def train_baseline_seasonal_naive(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple:
    """Train a seasonal naive baseline model.

    Seasonal naive baseline: predict tomorrow = same hour last week (lag-168).
    The prediction for each row is the value from 168 hours ago.

    Args:
        X_train: Training features (unused — seasonal naive doesn't use features).
        y_train: Training target values.
        X_val: Validation features (unused).
        y_val: Validation target values.

    Returns:
        A tuple ``(model, metrics)`` where model is a fitted
        ``SeasonalNaiveModel`` and metrics is a dict.
    """
    model = SeasonalNaiveModel(season_length=168)
    model.fit(X_train, y_train)
    y_pred = np.asarray(model.predict(X_val))
    metrics = compute_metrics(y_val, y_pred, ["rmse", "mae"])
    return model, metrics


def log_to_mlflow(
    model,
    metrics: dict[str, float],
    cfg: dict,
    feature_names: list[str] | None = None,
    run_name: str | None = None,
    promote_to_production: bool = False,
) -> str:
    """Log model, params, and metrics to MLflow.

    Logs:
    - Model parameters (from ``cfg["model"]["params"]``)
    - Model type and feature flags
    - Temporal config (resolution, horizon)
    - Metrics (rmse, mae, mape, r2, etc.)
    - Feature importances as a JSON artifact (if available)
    - Full ``params.yaml`` as an artifact (for reproducibility)
    - Tags: ``params_hash`` (SHA256 fingerprint), ``git_commit``, ``stage``

    If ``promote_to_production`` is True and the model was registered, the
    new version is set as the champion alias (``mlflow.champion_alias``).
    Baselines must NOT be promoted — only the primary model becomes the
    deployment target. The promotion is additionally gated by
    ``mlflow.promote_to_production`` in the config.

    Args:
        model: A fitted scikit-learn-compatible model.
        metrics: Dict of metric name to value.
        cfg: Configuration dict with MLflow and model settings.
        feature_names: List of feature column names (for feature importances).
        run_name: Optional name for the MLflow run.
        promote_to_production: Whether to set the champion alias on the
            newly registered version. Defaults to False (safe default —
            baselines are registered but never promoted).

    Returns:
        The MLflow run ID as a string.
    """
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    model_name = cfg["mlflow"]["model_name"]

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log model parameters
        for key, value in cfg["model"]["params"].items():
            mlflow.log_param(key, value)

        # Log model type
        mlflow.log_param("model_type", cfg["model"]["type"])

        # Log feature flags
        for group, settings in cfg["features"].items():
            if isinstance(settings, dict) and "enabled" in settings:
                mlflow.log_param(f"feature_{group}", settings["enabled"])

        # Log temporal config
        mlflow.log_param("resolution", cfg["temporal"]["resolution"])
        mlflow.log_param("horizon", cfg["temporal"]["horizon"])

        # Log params hash for quick run comparison
        params_hash = compute_params_hash()
        mlflow.set_tag("params_hash", params_hash)

        # Log full params.yaml as artifact for reproducibility
        params_path = Path("params.yaml")
        if params_path.exists():
            mlflow.log_artifact(str(params_path), artifact_path="config")

        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Log model artifact. Use cloudpickle serialization: MLflow 3.x
        # defaults to skops, which refuses to serialize XGBoost models and
        # the custom baseline classes unless they are explicitly whitelisted.
        model_info = log_model(
            sk_model=model,
            name="model",
            registered_model_name=model_name,
            serialization_format="cloudpickle",
        )

        # Promote to champion alias (only the primary model; baselines are
        # registered but never promoted). An alias points to exactly one
        # version, so multiple promotions cannot accumulate the way registry
        # stages did.
        promote = promote_to_production and cfg["mlflow"].get(
            "promote_to_production", True
        )
        if promote and model_info.registered_model_version:
            client = mlflow.MlflowClient()
            alias = cfg["mlflow"].get("champion_alias", "champion")
            client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=str(model_info.registered_model_version),
            )
            logger.info(
                "Model version %s set as '%s' alias",
                model_info.registered_model_version,
                alias,
            )
        else:
            logger.debug(
                "Model version %s registered without promotion "
                "(promote_to_production=%s, config=%s)",
                model_info.registered_model_version,
                promote_to_production,
                cfg["mlflow"].get("promote_to_production", True),
            )

        # Log feature importances if available
        if feature_names is not None:
            log_feature_importances(model, feature_names)

        # Log tags
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.set_tag("stage", "2")

        logger.debug("MLflow tracking URI: %s", cfg["mlflow"]["tracking_uri"])
        logger.debug("MLflow experiment: %s", cfg["mlflow"]["experiment_name"])
        logger.info("MLflow run ID: %s", run_id)

    return run_id


def main():
    """Orchestrate training pipeline: train XGBoost + baselines, log to MLflow."""
    cfg = load_config()
    setup_logging(cfg, logger_name=MODULE_LOGGER_NAME)
    processed_path = cfg["data"]["processed_path"]

    logger.info("Stage: training")
    logger.info("Loading features")
    X_train, y_train, X_val, y_val, X_test, y_test = load_features(processed_path, cfg)
    feature_names = get_feature_names(processed_path, cfg)

    # --- Train XGBoost ---
    logger.info("Loading model: %s", cfg["model"]["type"])
    model = load_model(cfg)
    logger.debug("Model: %s", model)

    logger.info("Training XGBoost")
    model.fit(X_train, y_train)
    logger.info("Training complete")

    logger.info("Evaluating on validation set")
    y_pred = np.asarray(model.predict(X_val))
    metrics = compute_metrics(y_val, y_pred, cfg["evaluation"]["metrics"])
    for name, value in metrics.items():
        logger.info("Validation %s: %.4f", name, value)

    logger.info("Logging XGBoost to MLflow")
    # Primary model: promoted to the champion alias
    log_to_mlflow(
        model,
        metrics,
        cfg,
        feature_names=feature_names,
        run_name="xgboost-v1",
        promote_to_production=True,
    )

    # --- Train baselines ---
    # Baselines are registered for comparison but NEVER promoted
    # (promote_to_production defaults to False).
    logger.info("Training baseline: persistence")
    persistence_model, persistence_metrics = train_baseline_persistence(
        X_train, y_train, X_val, y_val
    )
    for name, value in persistence_metrics.items():
        logger.info("Persistence baseline validation %s: %.4f", name, value)

    logger.info("Logging persistence baseline to MLflow")
    log_to_mlflow(
        persistence_model,
        persistence_metrics,
        cfg,
        run_name="baseline-persistence",
    )

    logger.info("Training baseline: seasonal naive")
    seasonal_model, seasonal_metrics = train_baseline_seasonal_naive(
        X_train, y_train, X_val, y_val
    )
    for name, value in seasonal_metrics.items():
        logger.info("Seasonal naive baseline validation %s: %.4f", name, value)

    logger.info("Logging seasonal naive baseline to MLflow")
    log_to_mlflow(
        seasonal_model,
        seasonal_metrics,
        cfg,
        run_name="baseline-seasonal-naive",
    )

    # --- Summary ---
    logger.info(
        "Model comparison (validation RMSE): xgboost=%.4f, "
        "persistence=%.4f, seasonal_naive=%.4f",
        metrics["rmse"],
        persistence_metrics["rmse"],
        seasonal_metrics["rmse"],
    )

    logger.info("Training complete")


if __name__ == "__main__":
    main()
