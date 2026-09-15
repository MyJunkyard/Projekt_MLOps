"""
features/derivatives.py — Derivative (difference) features of the target.

Holds the existing ``add_derivative_features`` moved verbatim from
``featurise.py`` (Workstream 0 module restructure). Further time
derivatives and derived signals are planned for Stage 4 (EXP-008+);
new code of this role belongs here.
"""

import pandas as pd


def add_derivative_features(
    df: pd.DataFrame,
    target_col: str,
    order: list[int] | None = None,
    smooth_window: int = 3,
) -> pd.DataFrame:
    """Add derivative (difference) features of the target variable.

    Raw price derivatives amplify noise. Before computing:
    1. Apply a short rolling mean (window=smooth_window) to smooth
    2. Then compute diff(1) for first derivative, diff(2) for second

    Conceptually: first derivative = price velocity (rising/falling)
    Second derivative = price acceleration (speeding up/slowing down)

    Args:
        df: DataFrame containing ``target_col``, sorted by ``timestamp``
            (differences are order-dependent).
        target_col: Name of the target column.
        order: List of derivative orders to compute (1, 2, or both).
        smooth_window: Rolling mean window before differencing.

    Returns:
        The same DataFrame (mutated in place and returned) with float
        columns ``{target_col}_diff_{n}`` added for each order ``n``.
    """
    # Smooth the target to reduce noise
    if order is None:
        order = [1, 2]
    smoothed = df[target_col].rolling(window=smooth_window, min_periods=1).mean()

    for n in order:
        df[f"{target_col}_diff_{n}"] = smoothed.diff(n)

    return df
