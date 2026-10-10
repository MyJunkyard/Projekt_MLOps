"""features/derivatives.py — Derivative (difference) features of the target.

Raw price derivatives amplify noise, so the target is smoothed with a short
rolling mean before differencing. Because the useful smoothing horizon is not
one-size-fits-all, WS0 made the window a *list*: every ``(order, window)``
pair yields its own column, and the run book selects the variant that works
via ``features.select`` (EXP-008/009).

Naming is ``{target_col}_diff_{order}_w{window}`` — the window suffix is
mandatory, never implied. A bare ``price_diff_1`` would be ambiguous the
moment a second window exists, so it is never produced.
"""

import pandas as pd


def make_derivative_name(target_col: str, order: int, window: int) -> str:
    """Build the derivative column name for one ``(order, window)`` pair.

    Args:
        target_col: Name of the target column.
        order: Difference order (1 = velocity, 2 = acceleration).
        window: Rolling-mean window applied before differencing.

    Returns:
        The column name, e.g. ``price_eur_mwh_diff_1_w3``.
    """
    return f"{target_col}_diff_{order}_w{window}"


def add_derivative_features(
    df: pd.DataFrame,
    target_col: str,
    order: list[int] | None = None,
    smooth_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add derivative (difference) features of the target variable.

    Conceptually: first derivative = price velocity (rising/falling),
    second derivative = price acceleration (speeding up/slowing down). Each
    smoothing window is a separate variant, so a config asking for orders
    ``[1, 2]`` and windows ``[3, 6]`` produces four columns.

    Args:
        df: DataFrame containing ``target_col``, sorted by ``timestamp``
            (differences are order-dependent).
        target_col: Name of the target column.
        order: Derivative orders to compute (1, 2, or both). Defaults to
            ``[1, 2]``.
        smooth_windows: Rolling-mean windows to apply before differencing,
            one column per window. Defaults to ``[3]``.

    Returns:
        The same DataFrame (mutated in place and returned) with float columns
        ``{target_col}_diff_{n}_w{w}`` for every order/window pair.
    """
    if order is None:
        order = [1, 2]
    if smooth_windows is None:
        smooth_windows = [3]

    for window in smooth_windows:
        smoothed = df[target_col].rolling(window=window, min_periods=1).mean()
        for n in order:
            df[make_derivative_name(target_col, n, window)] = smoothed.diff(n)

    return df
