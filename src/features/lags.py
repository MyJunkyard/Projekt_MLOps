"""
features/lags.py — Lag and rolling features of the target variable.

Extracted verbatim from ``featurise.py`` (Workstream 0 module restructure).
"""

import pandas as pd


def add_lag_features(
    df: pd.DataFrame, periods: list[int]
) -> pd.DataFrame:
    """Add lag features of the target variable.

    Lags are computed on the full dataset before splitting to avoid
    data leakage. Rows with NaN from lag creation at the beginning
    of the series are dropped.

    Args:
        df: DataFrame with the target column ``price_eur_mwh``, sorted
            by ``timestamp`` (lags are order-dependent).
        periods: Lag periods in hours, e.g. [1, 2, 3, 24, 48, 168].

    Returns:
        The same DataFrame (mutated in place and returned) with float
        lag columns ``lag_{p}h`` added (NaN for the first ``p`` rows).
    """
    for period in periods:
        df[f"lag_{period}h"] = df["price_eur_mwh"].shift(period)
    return df


def add_rolling_features(
    df: pd.DataFrame, target_col: str, windows: list[int] | None = None
) -> pd.DataFrame:
    """Add rolling statistics of the target variable.

    Rolling features are computed on the full dataset before splitting to
    avoid data leakage. Uses trailing windows (no future data).

    Args:
        df: DataFrame containing ``target_col``, sorted by ``timestamp``
            (rolling windows are order-dependent).
        target_col: Name of the target column to compute rolling stats on.
        windows: List of window sizes in hours.

    Returns:
        The same DataFrame (mutated in place and returned) with float
        columns ``rolling_mean_{w}h`` and ``rolling_std_{w}h`` added for
        each window ``w``.
    """
    if windows is None:
        windows = [24, 168]
    for window in windows:
        df[f"rolling_mean_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).std()
        )

    return df
