"""
training/baselines.py — Baseline models and their training helpers.

Moved verbatim from ``train.py`` (Workstream 0 module restructure).

**Cloudpickle caveat:** these classes are cloudpickled *by reference*
into MLflow; their module path is part of the pickle. Moving them
invalidates model artifacts from old runs — re-run ``make train`` in the
same commit as this move.
"""

import numpy as np
import numpy.typing as npt

from src.common.metrics import compute_metrics


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
