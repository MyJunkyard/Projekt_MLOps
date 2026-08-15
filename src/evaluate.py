"""
evaluate.py — Stage 1: Evaluate trained model on test set.

Loads model from MLflow registry, computes metrics on test split,
and prints results to stdout. Stub version — residual plots added in Stage 4.
"""

import mlflow
import numpy as np
import pandas as pd

from src.utils import compute_metrics, get_split_masks, load_config


def load_model_from_registry(cfg: dict):
    """Load model from MLflow model registry.

    Uses the model stage specified in params.yaml (default: Production).

    Args:
        cfg: Configuration dict with ``mlflow.tracking_uri``,
            ``mlflow.model_name``, and ``serving.model_stage``.

    Returns:
        A loaded MLflow pyfunc model.
    """
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    model_uri = f"models:/{cfg['mlflow']['model_name']}/{cfg['serving']['model_stage']}"
    print(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def load_test_features(path: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load test split from features.parquet.

    Args:
        path: Path to the features Parquet file.
        cfg: Configuration dict with ``data.target_col`` and split dates.

    Returns:
        A tuple ``(X_test, y_test)`` of numpy arrays.
    """
    df = pd.read_parquet(path)

    target_col = cfg["data"]["target_col"]
    _, _, test_mask = get_split_masks(df, cfg)
    df_test = df.iloc[test_mask]

    y_test = df_test[target_col].values
    feature_cols = [c for c in df_test.columns if c not in [target_col, "timestamp"]]
    X_test = df_test[feature_cols].values

    print(f"Test set: {len(df_test):,} rows, {len(feature_cols)} features")
    return X_test, y_test


def print_results_table(metrics: dict[str, float]) -> None:
    """Print metrics in a formatted table.

    Args:
        metrics: Dict of metric name to value.
    """
    print("\n" + "=" * 40)
    print("  Test Set Evaluation Results")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"  {name.upper():>8}: {value:.4f}")
    print("=" * 40)


def main():
    """Orchestrate evaluation."""
    cfg = load_config()
    processed_path = cfg["data"]["processed_path"]

    print("=== Loading test features ===")
    X_test, y_test = load_test_features(processed_path, cfg)

    print("\n=== Loading model from registry ===")
    model = load_model_from_registry(cfg)

    print("\n=== Computing predictions ===")
    y_pred = model.predict(X_test)

    print("\n=== Computing metrics ===")
    metrics = compute_metrics(y_test, y_pred, cfg["evaluation"]["metrics"])
    print_results_table(metrics)

    print("\nDone.")
    # TODO: add residual breakdown plots in Stage 4


if __name__ == "__main__":
    main()
