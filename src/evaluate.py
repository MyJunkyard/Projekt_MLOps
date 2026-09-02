"""
evaluate.py — Stage 2: Evaluate trained model with metrics, plots, and
residual analysis.

Loads model from MLflow registry, computes metrics on test split, generates
actual-vs-predicted plots and residual breakdowns, and logs them as MLflow
artifacts. Gated by ``evaluation.generate_plots`` in params.yaml.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.utils import compute_metrics, get_split_masks, load_config, setup_logging

# Stable module name (not `__name__` — under `python -m` it is
# `"__main__"` and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.evaluate"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def load_model_from_registry(cfg: dict):
    """Load model from MLflow model registry.

    Uses the model alias specified in params.yaml (default: champion).
    An alias points to exactly one registered version, so the loaded model
    is unambiguous.

    Args:
        cfg: Configuration dict with ``mlflow.tracking_uri``,
            ``mlflow.model_name``, and ``serving.model_alias``.

    Returns:
        A loaded MLflow pyfunc model.
    """
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    alias = cfg["serving"]["model_alias"]
    model_uri = f"models:/{cfg['mlflow']['model_name']}@{alias}"
    logger.info("Loading model from %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def get_model_run_id(cfg: dict) -> str | None:
    """Resolve the MLflow run ID that produced the champion model version.

    Uses the model alias from ``serving.model_alias`` to look up the
    registered version and returns the run that created it, so evaluation
    results can be attached to the model's own run page.

    Args:
        cfg: Configuration dict with ``mlflow.tracking_uri``,
            ``mlflow.model_name``, and ``serving.model_alias``.

    Returns:
        The training run ID, or None if it cannot be resolved (e.g. the
        registry is unavailable or the alias does not exist yet).
    """
    try:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(
            cfg["mlflow"]["model_name"], cfg["serving"]["model_alias"]
        )
        return version.run_id
    except Exception as e:
        logger.warning("Could not resolve model run ID: %s", e)
        return None


def log_evaluation_results(
    cfg: dict, metrics: dict[str, float], artifact_paths: list[str]
) -> None:
    """Log evaluation metrics and artifacts to MLflow.

    Logs to the training run that produced the champion model version, so
    the results are reachable from the model's run page. Falls back to a
    new ``evaluation`` run when the training run cannot be resolved —
    logging outside any active run would silently create an orphan
    anonymous run.

    Args:
        cfg: Configuration dict with MLflow settings.
        metrics: Dict of metric name to value.
        artifact_paths: Paths of files to log as artifacts (may be empty).
    """
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    run_id = get_model_run_id(cfg)

    if run_id is not None:
        logger.info("Logging evaluation results to training run %s", run_id)
        with mlflow.start_run(run_id=run_id):
            _log_metrics_and_artifacts(metrics, artifact_paths)
    else:
        logger.warning(
            "Training run not resolved; logging evaluation results to a "
            "new 'evaluation' run"
        )
        with mlflow.start_run(run_name="evaluation"):
            _log_metrics_and_artifacts(metrics, artifact_paths)


def _log_metrics_and_artifacts(
    metrics: dict[str, float], artifact_paths: list[str]
) -> None:
    """Log metrics and artifact files to the active MLflow run."""
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    for path in artifact_paths:
        mlflow.log_artifact(path)
        logger.debug("Artifact logged: %s", path)


def load_test_features(path: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load test split from features.parquet.

    Input contract: the Parquet must contain the featurisation output
    schema — a tz-aware ``timestamp`` column, the ``data.target_col``
    target column, and one column per feature.

    Args:
        path: Path to the features Parquet file.
        cfg: Configuration dict with ``data.target_col`` and split dates
            (``data.train_end``, ``data.val_end``).

    Returns:
        A tuple ``(X_test, y_test)`` of numpy arrays: ``X_test`` of
        shape (n_test, n_features) and 1-D ``y_test`` of shape
        (n_test,).
    """
    df = pd.read_parquet(path)

    target_col = cfg["data"]["target_col"]
    _, _, test_mask = get_split_masks(df, cfg)
    df_test = df.iloc[test_mask]

    y_test = df_test[target_col].values
    feature_cols = [c for c in df_test.columns if c not in [target_col, "timestamp"]]
    X_test = df_test[feature_cols].values

    logger.info(
        "Test set: %s rows, %d features", f"{len(df_test):,}", len(feature_cols)
    )
    return X_test, y_test


def log_results_table(metrics: dict[str, float]) -> None:
    """Log test-set evaluation metrics line by line.

    Args:
        metrics: Dict of metric name to value.
    """
    for name, value in metrics.items():
        logger.info("Test %s: %.4f", name, value)


def plot_actual_vs_predicted(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    output_path: str,
) -> str:
    """Generate an actual-vs-predicted line plot and save as PNG.

    Args:
        y_true: Ground-truth target values. Array-like of shape
            (n_samples,); normalized to a 1-D float ndarray via
            ``np.asarray``.
        y_pred: Predicted target values. Array-like of the same length
            as ``y_true``; normalized the same way.
        output_path: Path to save the PNG file.

    Returns:
        The path the plot was saved to.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_true, label="Actual", alpha=0.8, linewidth=1.5)
    ax.plot(y_pred, label="Predicted", alpha=0.8, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title("Actual vs Predicted — Test Set")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path_obj, dpi=150)
    plt.close(fig)

    logger.info("Plot saved to %s", output_path_obj)
    return str(output_path_obj)


def residual_breakdown(
    df: pd.DataFrame,
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    cfg: dict,
) -> pd.DataFrame:
    """Compute mean absolute residual grouped by categorical variables.

    For each categorical variable in ``evaluation.residual_breakdown``
    (e.g. hour, day_of_week, month, is_holiday), this computes the mean
    absolute residual for each category value.

    Args:
        df: The original (test-split) DataFrame with the categorical
            grouping columns; must be row-aligned with ``y_true`` /
            ``y_pred`` (same order and length).
        y_true: Ground-truth target values. Array-like of shape
            (n_samples,); normalized to a 1-D float ndarray via
            ``np.asarray``.
        y_pred: Predicted target values. Array-like of the same length
            as ``y_true``; normalized the same way.
        cfg: Configuration dict with ``evaluation.residual_breakdown``
            (list of grouping column names).

    Returns:
        A DataFrame with columns ``grouping`` (str), ``category``,
        ``mean_abs_residual`` (float), and ``count`` (int).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = np.abs(y_true - y_pred)
    breakdown_cols = cfg["evaluation"]["residual_breakdown"]

    rows = []
    for col in breakdown_cols:
        if col not in df.columns:
            continue
        for category in sorted(df[col].unique()):
            mask = (df[col] == category).to_numpy()
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "grouping": col,
                    "category": category,
                    "mean_abs_residual": float(residuals[mask].mean()),
                    "count": int(mask.sum()),
                }
            )

    return pd.DataFrame(rows)


def main():
    """Orchestrate evaluation: metrics, plots, and residual breakdown."""
    cfg = load_config()
    setup_logging(cfg, logger_name=MODULE_LOGGER_NAME)
    processed_path = cfg["data"]["processed_path"]
    reports_dir = Path("reports")

    logger.info("Stage: evaluation")
    logger.info("Loading test features")
    df = pd.read_parquet(processed_path)
    X_test, y_test = load_test_features(processed_path, cfg)

    # Get test DataFrame columns for residual breakdown
    _, _, test_mask = get_split_masks(df, cfg)
    df_test = df.iloc[test_mask].reset_index(drop=True)

    logger.info("Loading model from registry")
    model = load_model_from_registry(cfg)

    logger.info("Computing predictions")
    y_pred = np.asarray(model.predict(X_test))

    logger.info("Computing metrics")
    metrics = compute_metrics(y_test, y_pred, cfg["evaluation"]["metrics"])
    log_results_table(metrics)

    artifact_paths: list[str] = []

    # Generate plots if enabled
    if cfg["evaluation"].get("generate_plots", False):
        reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating plots")
        plot_path = plot_actual_vs_predicted(
            y_test, y_pred, str(reports_dir / "actual_vs_predicted.png")
        )

        logger.info("Computing residual breakdown")
        breakdown_df = residual_breakdown(df_test, y_test, y_pred, cfg)

        logger.debug("Residual breakdown:\n%s", breakdown_df.to_string(index=False))

        # Save residual breakdown to CSV
        breakdown_path = reports_dir / "residual_breakdown.csv"
        breakdown_df.to_csv(breakdown_path, index=False)
        logger.info("Residual breakdown saved to %s", breakdown_path)

        artifact_paths = [plot_path, str(breakdown_path)]

    # Log metrics and artifacts to MLflow, attached to the training run
    # that produced the model (logging outside an active run would
    # silently create an orphan anonymous run)
    try:
        log_evaluation_results(cfg, metrics, artifact_paths)
    except Exception as e:
        logger.warning("Could not log results to MLflow: %s", e)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
