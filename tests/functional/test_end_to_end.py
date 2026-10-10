"""
Functional tests — full pipeline smoke test.

Runs the complete pipeline (ingest → featurise → train → evaluate) on a
small synthetic dataset using temporary files and mocked MLflow.
"""

from unittest import mock

import numpy as np

from src.common.metrics import compute_metrics
from src.common.splits import get_split_masks
from src.common.superset import build_manifest, superset_paths, write_manifest
from src.features.calendar import _get_holiday_dates, add_calendar_features
from src.features.lags import add_lag_features
from src.features.main import load_raw_data, save_reference_data
from src.ingestion.entsoe import generate_synthetic_data
from src.ingestion.manifest import save_raw_data
from src.ingestion.validation import validate_schema
from src.training.loader import load_model, load_superset_features


def _materialise(cfg, tmp_path, loaded):
    """Write the WS0 superset artifacts and return their directory.

    Replaces the old "save one ``features.parquet``" step: training and
    evaluation now read a superset (parquet + schema sidecar + manifest)
    whose columns carry the group metadata ``features.select`` needs.

    Args:
        cfg: Pipeline config (``data.superset_dir`` is redirected to tmp).
        tmp_path: Pytest temp directory.
        loaded: Fully engineered frame (NaNs may remain).

    Returns:
        The superset directory path.
    """
    from src.common.schema import schema_from_dataframe

    directory = tmp_path / "superset"
    directory.mkdir(parents=True, exist_ok=True)
    cfg.data.superset_dir = str(directory) + "/"

    paths = superset_paths(cfg)
    loaded.to_parquet(paths.parquet, index=False)
    schema_from_dataframe(loaded, target_col=cfg.data.target_col).save(paths.schema)
    write_manifest(paths, build_manifest(paths, cfg))
    return directory


class TestEndToEndPipeline:
    @mock.patch("src.training.registry.mlflow")
    def test_full_pipeline_smoke(self, mock_mlflow, tmp_path, sample_config):
        """Positive: full pipeline runs end-to-end and produces valid outputs."""
        # generate_synthetic_data starts at 2020-01-01; align split dates to it
        cfg = sample_config
        cfg.data.train_end = "2020-01-05"
        cfg.data.val_end = "2020-01-10"

        # --- 1. Ingest ---
        df = generate_synthetic_data(n_hours=300, seed=42)
        validate_schema(df)
        raw_path = tmp_path / "raw" / "synthetic.csv"
        save_raw_data(df, str(raw_path))

        # --- 2. Featurise (superset materialisation, NaNs kept) ---
        loaded = load_raw_data(str(raw_path))
        loaded = loaded.sort_values("timestamp").reset_index(drop=True)
        holidays = _get_holiday_dates(loaded)
        loaded = add_calendar_features(loaded, holidays)
        loaded = add_lag_features(loaded, cfg.data.target_col, [1, 2, 24])

        superset_dir = _materialise(cfg, tmp_path, loaded)

        # --- 3. Train ---
        X_splits, y_splits, _, _, clean = load_superset_features(cfg)
        train_mask, _, _ = get_split_masks(clean, cfg.data)
        X_train, y_train = X_splits[0], y_splits[0]

        # Reference dataset: train split of the clean superset, so the drift
        # baseline is exactly the rows the model was trained on.
        reference = tmp_path / "reference" / "reference.parquet"
        reference.parent.mkdir(parents=True, exist_ok=True)
        save_reference_data(clean.iloc[train_mask].copy(), str(reference))

        model = load_model(cfg.model)
        model.fit(X_train, y_train)

        # --- 4. Evaluate ---
        y_pred = model.predict(X_splits[2])
        y_test = y_splits[2]
        metrics = compute_metrics(y_test, y_pred, cfg.evaluation.metrics)

        # --- Assertions ---
        assert superset_dir.exists()
        assert reference.exists()
        assert y_pred.shape == y_test.shape
        assert np.isfinite(y_pred).all()
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0

        # Model is a fitted DummyRegressor predicting the mean
        assert np.allclose(y_pred, y_train.mean(), atol=1e-6)

    @mock.patch("src.training.registry.mlflow")
    def test_full_pipeline_with_xgboost(
        self, mock_mlflow, tmp_path, sample_config_stage2
    ):
        """Positive: full pipeline with XGBoost produces finite predictions."""
        cfg = sample_config_stage2
        cfg.data.train_end = "2020-01-05"
        cfg.data.val_end = "2020-01-10"

        # --- 1. Ingest ---
        df = generate_synthetic_data(n_hours=300, seed=42)
        validate_schema(df)
        raw_path = tmp_path / "raw" / "synthetic.csv"
        save_raw_data(df, str(raw_path))

        # --- 2. Featurise (superset materialisation, NaNs kept) ---
        loaded = load_raw_data(str(raw_path))
        loaded = loaded.sort_values("timestamp").reset_index(drop=True)
        holidays = _get_holiday_dates(loaded)
        loaded = add_calendar_features(loaded, holidays)
        loaded = add_lag_features(loaded, cfg.data.target_col, [1, 2, 24])

        superset_dir = _materialise(cfg, tmp_path, loaded)

        # Derive the reference baseline and the training split from the same
        # clean superset frame the training stage reads, so the drift baseline
        # is exactly the rows the model was trained on.
        X_splits, y_splits, _, _, clean = load_superset_features(cfg)
        train_mask, _, _ = get_split_masks(clean, cfg.data)
        X_train, y_train = X_splits[0], y_splits[0]

        # Reference dataset: train split of the clean superset, so the drift
        # baseline is exactly the rows the model was trained on.
        reference = tmp_path / "reference" / "reference.parquet"
        reference.parent.mkdir(parents=True, exist_ok=True)
        save_reference_data(clean.iloc[train_mask].copy(), str(reference))

        # --- 3. Train XGBoost (reads the superset) ---
        model = load_model(cfg.model)
        model.fit(X_train, y_train)

        # --- 4. Evaluate ---
        y_pred = model.predict(X_splits[2])
        y_test = y_splits[2]
        metrics = compute_metrics(y_test, y_pred, cfg.evaluation.metrics)

        # --- Assertions ---
        assert superset_dir.exists()
        assert reference.exists()
        assert y_pred.shape == y_test.shape
        assert np.isfinite(y_pred).all()
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0

        # Model is a fitted XGBoostRegressor with sensible predictions
        assert len(y_pred) == len(y_test)
