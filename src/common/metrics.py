"""
common/metrics.py — Regression metric computation (Stage-4 metric set).

Rewritten in WS7 step 1 for the Stage-4 metric contract
(``docs/metrics_research.md``; ``docs/stage4_ws7_implementation_plan.md``
§4.1). The Stage-2/3 implementation — including MAPE and its
``y_true != 0`` mask — is gone (parent-plan D9): MAPE explodes on the
near-zero and negative electricity prices this pipeline scores, and no
window guard can rescue it.

Contract (WS7 local D1/D2):

- ``compute_metrics(y_true, y_pred, metrics_list, *, y_train=None,
  naive_period=None)`` — the keyword-only additions are backward
  compatible; existing positional call sites keep working.
- Unknown names (``mape`` included) **raise** ``ValueError`` instead of
  being silently skipped — a typo must never read as "metric omitted".
- ``rmae`` requires ``naive_period``; ``mase`` requires ``y_train`` *and*
  ``naive_period``; a window no longer than the period raises instead of
  emitting ``NaN`` (parent G9).
- Rows are never dropped: sMAPE floors the *denominator* only (parent
  G8), so near-zero targets stay finite instead of being masked out.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#: Metric names this module implements. The vocabulary source of truth is
#: the ``SupportedMetric`` Literal in ``src/config/models.py``; the
#: parametrized contract test ``TestSupportedMetricContract`` pins the
#: two together, so this set may never drift from the Literal (or from
#: ``compute_metrics`` itself).
SUPPORTED: frozenset[str] = frozenset(
    {"mae", "rmse", "smape", "rmae", "mase", "r2"}
)

#: Denominator floor for sMAPE. Keeps the metric finite when both
#: ``y_true`` and ``y_pred`` are ~0 without masking any row (G8).
_EPS = 1e-12


def compute_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    metrics_list: Sequence[str],
    *,
    y_train: np.ndarray | list[float] | None = None,
    naive_period: int | None = None,
) -> dict[str, float]:
    """Compute requested regression metrics and return as a dict.

    Args:
        y_true: Ground-truth target values. Array-like of shape
            (n_samples,); normalized to a 1-D float ndarray via
            ``np.asarray``. This is the *evaluation window* (val or test
            split) — the rMAE denominator is computed on the same window.
        y_pred: Predicted target values. Array-like of shape
            (n_samples,); normalized the same way.
        metrics_list: Names of metrics to compute. Supported values:
            ``"mae"``, ``"rmse"``, ``"smape"``, ``"rmae"``, ``"mase"``,
            ``"r2"`` (e.g. ``evaluation.metrics`` from the config).
            Unsupported names — ``"mape"`` included — raise.
        y_train: In-sample training targets, required only by ``mase``
            (its denominator is the seasonal-naive MAE over the training
            window, per Hyndman & Koehler 2006).
        naive_period: Season length in native time steps, required by
            ``rmae`` and ``mase``. Comes from
            ``resolve_naive_period``/``evaluation.naive_period`` — this
            module is resolution-agnostic and treats it as a plain int.

    Returns:
        A dict mapping each requested metric name to its float value.

    Raises:
        ValueError: If any requested name is not in ``SUPPORTED`` (the
            message names the supported set and why ``mape`` is gone); if
            ``rmae`` is requested without ``naive_period``; if ``mase`` is
            requested without ``y_train``/``naive_period``; or if a
            window is no longer than ``naive_period`` (or its seasonal
            denominator is exactly 0), where the metric would otherwise
            be undefined (parent G9 — actionable error, never ``NaN``).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    unknown = [m for m in metrics_list if m not in SUPPORTED]
    if unknown:
        raise ValueError(
            f"Unsupported metric(s) {unknown}. Supported metrics: "
            f"{sorted(SUPPORTED)}. Note: 'mape' was removed in WS7 — it "
            "explodes on near-zero/negative electricity prices "
            "(docs/metrics_research.md §2)."
        )

    results = {}
    for metric in metrics_list:
        if metric == "rmse":
            results["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif metric == "mae":
            results["mae"] = float(mean_absolute_error(y_true, y_pred))
        elif metric == "r2":
            results["r2"] = float(r2_score(y_true, y_pred))
        elif metric == "smape":
            # Symmetric MAPE (Armstrong/Flores): bounded in [0, 200],
            # sign-safe, and — with the epsilon in the DENOMINATOR only —
            # finite for zero/near-zero targets without dropping rows.
            denominator = np.abs(y_true) + np.abs(y_pred) + _EPS
            results["smape"] = float(
                100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / denominator)
            )
        elif metric == "rmae":
            if naive_period is None:
                raise ValueError(
                    "rmae requires naive_period (the seasonal-naive lag in "
                    "native steps) — pass naive_period=... "
                    "(evaluation.naive_period in params.yaml)."
                )
            if len(y_true) <= naive_period:
                raise ValueError(
                    f"rmae requires an evaluation window longer than "
                    f"naive_period={naive_period} (got {len(y_true)} "
                    "rows) — the seasonal-naive denominator would be empty "
                    "(parent G9)."
                )
            # Denominator on the EVALUATION window (metrics_research §4.3):
            # the naive benchmark scored on the same rows as the model,
            # not a train-period constant.
            naive_mae = float(
                mean_absolute_error(y_true[naive_period:], y_true[:-naive_period])
            )
            if naive_mae == 0.0:
                raise ValueError(
                    f"rmae is undefined: the seasonal-naive MAE over the "
                    f"evaluation window is 0 (naive_period={naive_period})."
                )
            results["rmae"] = float(
                mean_absolute_error(y_true, y_pred) / naive_mae
            )
        elif metric == "mase":
            if y_train is None or naive_period is None:
                raise ValueError(
                    "mase requires y_train (the in-sample denominator "
                    "window) and naive_period — both are mandatory "
                    "(metrics_research §4.1)."
                )
            y_train_arr = np.asarray(y_train, dtype=float)
            if len(y_train_arr) <= naive_period:
                raise ValueError(
                    f"mase requires y_train longer than naive_period="
                    f"{naive_period} (got {len(y_train_arr)} rows) — the "
                    "in-sample denominator would be empty (parent G9)."
                )
            # In-sample seasonal-naive MAE: mean |y_t - y_{t-k}| over the
            # training window (lag-k differences, Hyndman-Koehler) — the
            # same expression as the rMAE denominator, over y_train.
            scale = float(
                np.mean(
                    np.abs(y_train_arr[naive_period:] - y_train_arr[:-naive_period])
                )
            )
            if scale == 0.0:
                raise ValueError(
                    f"mase is undefined: the in-sample seasonal-naive MAE "
                    f"is 0 (naive_period={naive_period})."
                )
            results["mase"] = float(mean_absolute_error(y_true, y_pred) / scale)
    return results
