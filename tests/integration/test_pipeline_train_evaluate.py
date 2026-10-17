"""
Integration tests — feature superset → train → evaluate flow.

Exercises the interaction between the training and evaluation packages
using temporary files and mocked MLflow (no live MLflow server).
"""

from unittest import mock

import numpy as np

from src.common.selection import drop_selected_nan_rows, open_superset
from src.common.splits import get_split_masks
from src.config.models import FeatureSelection
from src.evaluation.reporting import (
    load_test_features,
    resolve_evaluate_columns,
)
from src.training.baselines import (
    train_baseline_persistence,
    train_baseline_seasonal_naive,
)
from src.training.loader import (
    load_model,
    load_superset_features,
)

#: The fixture superset (``superset_dir``) keeps the warm-up NaNs of
#: ``sample_df_with_features``: ``lag_24h`` is NaN for the first 24 rows of
#: the 72-hour fixture — which is exactly its train window. With the default
#: all-column selection ``load_superset_features`` drops those rows *before*
#: splitting (per-selection dropna — behaviour asserted by
#: ``tests/unit/training/test_selection.py::TestPerSubsetNaN``) and the train
#: split comes out empty. Narrowing the selection to the NaN-free calendar
#: columns keeps the train window populated and exercises WS0's
#: ``features.select`` mechanism end to end.
CALENDAR_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "week_of_year",
    "is_holiday",
    "is_workday",
]


def _splits(cfg):
    """Load the superset and return the six train/val/test arrays.

    Thin adapter over ``load_superset_features`` (which returns the three
    splits as lists) so the tests read like the old 6-tuple API
    ``(X_train, y_train, X_val, y_val, X_test, y_test)``.
    """
    X_splits, y_splits, _, _, _ = load_superset_features(cfg)
    X_train, X_val, X_test = X_splits
    y_train, y_val, y_test = y_splits
    return X_train, y_train, X_val, y_val, X_test, y_test


class TestTrainEvaluateFlow:
    @mock.patch("src.evaluation.reporting.get_model_run_id", return_value=None)
    def test_train_then_evaluate(self, mock_run_id, superset_dir, sample_config):
        """Positive: load superset, train model, evaluate on test split."""
        sample_config.features.select = FeatureSelection(columns=CALENDAR_COLUMNS)
        X_train, y_train, X_val, y_val, _, _ = _splits(sample_config)
        model = load_model(sample_config.model)
        model.fit(X_train, y_train)

        # Model is fitted and can predict
        y_pred_val = model.predict(X_val)
        assert y_pred_val.shape == y_val.shape

        # --- Evaluate stage ---
        frame, schema = open_superset(sample_config)
        feature_cols, _ = resolve_evaluate_columns(sample_config, schema)
        df_clean = drop_selected_nan_rows(
            frame, feature_cols, sample_config.data.target_col
        )
        # Same test split the trainer saw: the mask is taken on the clean
        # frame and applied to it — mirrors evaluation.main().
        _, _, test_mask = get_split_masks(df_clean, sample_config.data)
        df_test = df_clean[test_mask]
        X_test_eval, y_test_eval = load_test_features(
            df_test, feature_cols, sample_config.data.target_col
        )
        y_pred_test = model.predict(X_test_eval)
        assert y_pred_test.shape == y_test_eval.shape
        assert np.isfinite(y_pred_test).all()

    @mock.patch("src.evaluation.reporting.get_model_run_id", return_value=None)
    def test_train_evaluate_consistent_splits(
        self, mock_run_id, superset_dir, sample_config
    ):
        """Positive: train and evaluate use the same test split."""
        sample_config.features.select = FeatureSelection(columns=CALENDAR_COLUMNS)
        *_, X_test_train, y_test_train = _splits(sample_config)

        frame, schema = open_superset(sample_config)
        feature_cols, _ = resolve_evaluate_columns(sample_config, schema)
        df_clean = drop_selected_nan_rows(
            frame, feature_cols, sample_config.data.target_col
        )
        _, _, test_mask = get_split_masks(df_clean, sample_config.data)
        df_test = df_clean[test_mask]
        X_test_eval, y_test_eval = load_test_features(
            df_test, feature_cols, sample_config.data.target_col
        )
        np.testing.assert_array_equal(X_test_train, X_test_eval)
        np.testing.assert_array_equal(y_test_train, y_test_eval)

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_log_to_mlflow_integration(
        self, mock_mlflow, mock_log_model, superset_dir, sample_config
    ):
        """Positive: full train + log_to_mlflow with mocked MLflow."""
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "integration-run"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.MlflowClient.return_value = mock.MagicMock()
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)

        from src.training.registry import log_to_mlflow

        sample_config.features.select = FeatureSelection(columns=CALENDAR_COLUMNS)
        X_train, y_train, X_val, y_val, _, _ = _splits(sample_config)
        model = load_model(sample_config.model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        from src.common.metrics import compute_metrics

        metrics = compute_metrics(y_val, y_pred, sample_config.evaluation.metrics)
        run_id = log_to_mlflow(model, metrics, sample_config)
        assert run_id == "integration-run"
        mock_mlflow.log_metric.assert_any_call("rmse", metrics["rmse"])


class TestXGBoostVsBaselines:
    def test_xgboost_produces_finite_predictions(
        self, superset_dir, sample_config_stage2
    ):
        """Positive: XGBoost fit + predict succeeds with finite metrics."""
        sample_config_stage2.data.superset_dir = superset_dir
        sample_config_stage2.features.select = FeatureSelection(
            columns=CALENDAR_COLUMNS
        )
        X_train, y_train, X_val, y_val, _, _ = _splits(sample_config_stage2)

        model = load_model(sample_config_stage2.model)
        model.fit(X_train, y_train)

        # Model is fitted and can predict
        y_pred_val = model.predict(X_val)
        assert y_pred_val.shape == y_val.shape

        # Baselines also train + predict without error
        _, persistence_metrics = train_baseline_persistence(
            X_train, y_train, X_val, y_val
        )
        _, seasonal_metrics = train_baseline_seasonal_naive(
            X_train, y_train, X_val, y_val
        )

        assert np.isfinite(persistence_metrics["rmse"])
        assert np.isfinite(seasonal_metrics["rmse"])

    def test_baselines_produce_reasonable_metrics(
        self, superset_dir, sample_config_stage2
    ):
        """Positive: baselines train and produce finite metrics."""
        sample_config_stage2.data.superset_dir = superset_dir
        sample_config_stage2.features.select = FeatureSelection(
            columns=CALENDAR_COLUMNS
        )
        X_train, y_train, X_val, y_val, _, _ = _splits(sample_config_stage2)

        _, persistence_metrics = train_baseline_persistence(
            X_train, y_train, X_val, y_val
        )
        _, seasonal_metrics = train_baseline_seasonal_naive(
            X_train, y_train, X_val, y_val
        )

        assert np.isfinite(persistence_metrics["rmse"])
        assert np.isfinite(seasonal_metrics["rmse"])

    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_feature_importances_logged(
        self, mock_mlflow, mock_log_model, superset_dir, sample_config_stage2
    ):
        """Positive: feature importances JSON artifact is logged."""
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "integration-run-2"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.MlflowClient.return_value = mock.MagicMock()
        mock_log_model.return_value = mock.MagicMock(registered_model_version=None)

        from src.training.registry import log_to_mlflow

        sample_config_stage2.data.superset_dir = superset_dir
        sample_config_stage2.features.select = FeatureSelection(
            columns=CALENDAR_COLUMNS
        )
        X_train, y_train, X_val, y_val, _, _ = _splits(sample_config_stage2)
        model = load_model(sample_config_stage2.model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        from src.common.metrics import compute_metrics

        metrics = compute_metrics(
            y_val, y_pred, sample_config_stage2.evaluation.metrics
        )
        log_to_mlflow(model, metrics, sample_config_stage2)

        # Verify feature importances were logged
        assert mock_mlflow.log_artifact.called


class TestOnlyPrimaryModelPromoted:
    @mock.patch("src.training.main.load_config")
    @mock.patch("src.training.registry.log_model")
    @mock.patch("src.training.registry.mlflow")
    def test_only_primary_model_promoted(
        self,
        mock_mlflow,
        mock_log_model,
        mock_load_config,
        superset_dir,
        sample_config_stage2,
    ):
        """Positive: main() sets the champion alias exactly once (xgboost only)."""
        from src.training.main import main as train_main

        mock_run = mock.MagicMock()
        mock_run.info.run_id = "integration-run-promote"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_log_model.return_value = mock.MagicMock(registered_model_version="1")

        sample_config_stage2.data.superset_dir = superset_dir
        sample_config_stage2.features.select = FeatureSelection(
            columns=CALENDAR_COLUMNS
        )
        # WS4: pin the val-split analysis OFF — this test pins promotion
        # behaviour only; the flag defaults to true in EvaluationConfig.
        sample_config_stage2.evaluation.run_analysis_after_train = False
        mock_load_config.return_value = sample_config_stage2
        train_main()

        # Three models are registered (xgboost + 2 baselines) but only the
        # primary model is promoted to the champion alias.
        client = mock_mlflow.MlflowClient.return_value
        alias_calls = client.set_registered_model_alias.call_args_list
        assert len(alias_calls) == 1
        call = alias_calls[0]
        assert call.kwargs["name"] == sample_config_stage2.mlflow.model_name
        assert call.kwargs["alias"] == "champion"
        assert call.kwargs["version"] == "1"
