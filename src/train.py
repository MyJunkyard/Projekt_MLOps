"""
train.py — Stage 1: Model training with dynamic loading and MLflow logging.

Loads features from Parquet, dynamically instantiates the model class
specified in params.yaml, trains on the train split, evaluates on val,
and logs everything to MLflow.
"""

import importlib
import subprocess

import mlflow
from mlflow.sklearn import log_model
import numpy as np
import pandas as pd

from src.utils import compute_metrics, get_split_masks, load_config


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

    print(f"Features: {len(feature_cols)} columns")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


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


def log_to_mlflow(model, metrics: dict[str, float], cfg: dict) -> str:
    """Log model, params, and metrics to MLflow.

    Args:
        model: A fitted scikit-learn-compatible model.
        metrics: Dict of metric name to value.
        cfg: Configuration dict with MLflow and model settings.

    Returns:
        The MLflow run ID as a string.
    """
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    model_name = cfg["mlflow"]["model_name"]

    with mlflow.start_run() as run:
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

        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Log model artifact
        model_info = log_model(
            sk_model=model,
            name="model",
            registered_model_name=model_name,
        )

        # Transition to Production stage
        # (Stage 1: single model; Stage 5+ uses Staging→Production promotion flow)
        if model_info.registered_model_version:
            client = mlflow.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=str(model_info.registered_model_version),
                stage="Production",
            )
            print(
                f"Model version {model_info.registered_model_version} "
                "transitioned to 'Production'."
            )

        # Log tags
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.set_tag("stage", "1")

        print(f"MLflow run ID: {run_id}")
        print(f"MLflow tracking URI: {cfg['mlflow']['tracking_uri']}")
        print(f"Experiment: {cfg['mlflow']['experiment_name']}")

    return run_id


def main():
    """Orchestrate training pipeline."""
    cfg = load_config()
    processed_path = cfg["data"]["processed_path"]

    print("=== Loading features ===")
    X_train, y_train, X_val, y_val, X_test, y_test = load_features(processed_path, cfg)

    print(f"\n=== Loading model: {cfg['model']['type']} ===")
    model = load_model(cfg)
    print(f"Model: {model}")

    print("\n=== Training ===")
    model.fit(X_train, y_train)
    print("Training complete.")

    print("\n=== Evaluating on validation set ===")
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val, y_pred, cfg["evaluation"]["metrics"])
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\n=== Logging to MLflow ===")
    log_to_mlflow(model, metrics, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
