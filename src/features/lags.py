"""
features/lags.py — Lag and rolling features of the target variable.

No-leakage contract (pinned by ``tests/unit/features/test_lags.py``):

- Lags are past-only ``shift(p)`` offsets: ``lag_{p}h`` of row N always
  equals the target of row N−p. Rolling statistics use trailing windows
  (``min_periods=1``), so they never see future rows.
- Both helpers run on the **full frame before** ``train_val_test_split``.
  Computing them per-split would leave NaN/wrong values at split starts
  and silently change the training data — the ordering is a regression
  test, not just a convention (see ``TestNoLeakageAcrossSplits``).

Naming note: the ``h`` suffix in ``lag_{p}h`` / ``rolling_*_{w}h``
assumes hourly resolution. Periods are native time steps; rescaling for
daily/weekly resolution is Stage 4 (EXP-012/013) scope.
"""

import pandas as pd

#: Fallback windows for direct library callers. The pipeline path always
#: passes ``features.lags.rolling_windows`` explicitly (see
#: ``features.main.build_features``); this constant only guards ad-hoc use.
DEFAULT_ROLLING_WINDOWS: tuple[int, int] = (24, 168)


def add_lag_features(
    df: pd.DataFrame, target_col: str, periods: list[int]
) -> pd.DataFrame:
    """Add lag features of the target variable.

    Args:
        df: DataFrame containing ``target_col``, sorted by ``timestamp``
            (lags are order-dependent).
        target_col: Name of the target column to lag.
        periods: Past offsets in native time steps, e.g. [1, 2, 3, 24,
            48, 168]. Duplicates are ignored; order is normalized.

    Returns:
        The same DataFrame (mutated in place and returned) with float
        lag columns ``lag_{p}h`` added (NaN for the first ``p`` rows).
    """
    for period in sorted(set(periods)):
        df[f"lag_{period}h"] = df[target_col].shift(period)
    return df


def add_rolling_features(
    df: pd.DataFrame, target_col: str, windows: list[int] | None = None
) -> pd.DataFrame:
    """Add rolling statistics of the target variable.

    Uses trailing windows (no future data): row N aggregates rows
    N−w+1 … N (fewer at the series start via ``min_periods=1``).

    Args:
        df: DataFrame containing ``target_col``, sorted by ``timestamp``
            (rolling windows are order-dependent).
        target_col: Name of the target column to compute rolling stats on.
        windows: Trailing window sizes in native time steps. ``None``
            falls back to ``DEFAULT_ROLLING_WINDOWS`` (direct-call
            convenience; the pipeline always passes config explicitly).
            Duplicates are ignored; order is normalized.

    Returns:
        The same DataFrame (mutated in place and returned) with float
        columns ``rolling_mean_{w}h`` and ``rolling_std_{w}h`` added for
        each window ``w``.
    """
    if windows is None:
        windows = list(DEFAULT_ROLLING_WINDOWS)
    for window in sorted(set(windows)):
        df[f"rolling_mean_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).std()
        )

    return df
