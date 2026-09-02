"""
Shared pytest fixtures for the MLOps test suite.

Fixtures provide small, deterministic, in-memory data and config so that
tests do not rely on real configuration files or data artifacts.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_config() -> dict:
    """Small in-memory config dict mirroring params.yaml structure."""
    return {
        "data": {
            "raw_path": "data/raw/",
            "processed_path": "data/processed/features.parquet",
            "reference_path": "data/reference/reference.parquet",
            "target_col": "price_eur_mwh",
            "date_col": "timestamp",
            "train_end": "2023-12-31",
            "val_end": "2024-01-01",
        },
        "temporal": {"resolution": "hourly", "horizon": 24},
        "features": {
            "calendar": {"enabled": True},
            "lags": {"enabled": True, "periods": [1, 2, 24]},
        },
        "model": {
            "type": "sklearn.dummy.DummyRegressor",
            "params": {"strategy": "mean"},
        },
        "evaluation": {"primary_metric": "rmse", "metrics": ["rmse", "mae"]},
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "energy-forecast",
            "model_name": "energy-forecast-model",
            "promote_to_production": True,
            "champion_alias": "champion",
        },
        "serving": {"port": 8000, "model_alias": "champion"},
    }


@pytest.fixture
def sample_config_stage2() -> dict:
    """Stage 2 config: XGBoost, lags enabled, holidays, full metrics."""
    return {
        "data": {
            "raw_path": "data/raw/",
            "processed_path": "data/processed/features.parquet",
            "reference_path": "data/reference/reference.parquet",
            "target_col": "price_eur_mwh",
            "date_col": "timestamp",
            "train_end": "2023-12-31",
            "val_end": "2024-01-01",
            "entsoe": {"bidding_zone": "PSE", "start_date": "2018-01-01"},
        },
        "temporal": {"resolution": "hourly", "horizon": 24},
        "features": {
            "calendar": {
                "enabled": True,
                "include": [
                    "hour",
                    "day_of_week",
                    "month",
                    "week_of_year",
                    "is_holiday",
                    "is_workday",
                    "days_to_next_holiday",
                    "days_since_last_holiday",
                ],
            },
            "lags": {"enabled": True, "periods": [1, 2, 24]},
            "derivatives": {"enabled": False, "order": [1, 2], "smooth_window": 3},
        },
        "model": {
            "type": "xgboost.XGBRegressor",
            "params": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
            },
        },
        "evaluation": {
            "primary_metric": "rmse",
            "metrics": ["rmse", "mae", "mape", "r2"],
            "generate_plots": True,
            "residual_breakdown": ["hour", "day_of_week", "month", "is_holiday"],
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "energy-forecast",
            "model_name": "energy-forecast-model",
            "promote_to_production": True,
            "champion_alias": "champion",
        },
        "serving": {"port": 8000, "model_alias": "champion"},
    }


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small hourly DataFrame spanning train/val/test boundaries.

    Spans 2023-12-30 00:00 to 2024-01-02 23:00 (72 hours) so that with
    train_end=2023-12-31 and val_end=2024-01-01 there is a non-empty
    train (24h), val (24h), and test (24h) split.
    """
    timestamps = pd.date_range(
        start="2023-12-30 00:00:00", periods=72, freq="h", tz="UTC"
    )
    prices = np.linspace(40.0, 60.0, num=72)
    return pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})


@pytest.fixture
def entsoe_sample_df() -> pd.DataFrame:
    """DataFrame mimicking ENTSO-E price data with gaps.

    Spans 2024-03-30 00:00 to 2024-04-01 23:00 (72 hours) with
    two rows removed to simulate data gaps.
    """
    timestamps = pd.date_range(
        start="2024-03-30 00:00:00", periods=72, freq="h", tz="UTC"
    )
    prices = np.linspace(40.0, 60.0, num=72)

    # Remove two rows to simulate gaps
    df = pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})
    df = df.drop(index=[26, 34]).reset_index(drop=True)
    return df


@pytest.fixture
def holiday_dates() -> list:
    """Known Polish public holiday dates for 2024."""
    return [
        "2024-01-01",  # New Year
        "2024-04-01",  # Easter Monday
        "2024-05-01",  # Labour Day
        "2024-05-03",  # Constitution Day
        "2024-05-30",  # Corpus Christi
        "2024-08-15",  # Assumption
        "2024-11-01",  # All Saints
        "2024-11-11",  # Independence Day
        "2024-12-25",  # Christmas Day
        "2024-12-26",  # Second Day of Christmas
    ]


@pytest.fixture
def sample_df_with_features(sample_df: pd.DataFrame) -> pd.DataFrame:
    """sample_df augmented with calendar and lag features."""
    df = sample_df.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_holiday"] = False
    df["is_workday"] = (df["day_of_week"] < 5) & (~df["is_holiday"])
    df["lag_1h"] = df["price_eur_mwh"].shift(1)
    df["lag_2h"] = df["price_eur_mwh"].shift(2)
    df["lag_24h"] = df["price_eur_mwh"].shift(24)
    return df


@pytest.fixture
def raw_csv_path(tmp_path, sample_df: pd.DataFrame) -> str:
    """Write sample_df to a temp CSV and return its path."""
    path = tmp_path / "raw" / "synthetic.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def features_parquet_path(tmp_path, sample_df_with_features: pd.DataFrame) -> str:
    """Write sample_df_with_features to a temp Parquet and return its path."""
    path = tmp_path / "processed" / "features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_df_with_features.to_parquet(path, index=False)
    return str(path)
