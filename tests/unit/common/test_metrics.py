"""
Unit tests for src/common/metrics.py — the Stage-4 metric set (WS7 step 1).

Pins the contract in docs/stage4_ws7_implementation_plan.md §4.1/§5:
sMAPE bounded and finite on zero targets, rMAE on the evaluation window,
MASE in-sample, actionable raises instead of NaN, and the removed MAPE
vocabulary failing loudly. The MAPE-era tests live in git history.
"""

import numpy as np
import pytest

from src.common.metrics import SUPPORTED, compute_metrics


class TestComputeMetrics:
    """Basic metrics + the Stage-4 additions."""

    @pytest.mark.parametrize(
        "metric,expected",
        [("rmse", 1.0), ("mae", 1.0), ("r2", -0.5)],
    )
    def test_metric_values(self, metric, expected):
        """Positive: known arrays produce hand-computed metric values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        result = compute_metrics(y_true, y_pred, [metric])
        assert result[metric] == pytest.approx(expected)

    def test_perfect_prediction_zero_errors(self):
        """Positive: perfect predictions yield zero error / full R²."""
        y_true = np.array([1.0, 2.0, 3.0])
        result = compute_metrics(y_true, y_true, ["rmse", "mae", "smape", "r2"])
        assert result["rmse"] == pytest.approx(0.0)
        assert result["mae"] == pytest.approx(0.0)
        assert result["smape"] == pytest.approx(0.0)
        assert result["r2"] == pytest.approx(1.0)


class TestSMAPE:
    """sMAPE: bounded in [0, 200], denominator-epsilon, no dropped rows (G8)."""

    def test_bounded_in_0_200(self):
        """Positive: sign-flipped forecasts sit at the 200 % ceiling."""
        y_true = np.array([1.0, -1.0, 5.0])
        y_pred = np.array([-1.0, 1.0, -5.0])
        value = compute_metrics(y_true, y_pred, ["smape"])["smape"]
        assert 0.0 <= value <= 200.0
        assert value == pytest.approx(200.0)

    def test_zero_target_finite(self):
        """Positive: y_true == 0 stays finite (epsilon in the denominator)."""
        y_true = np.array([0.0, 0.0, 10.0])
        y_pred = np.array([1.0, -2.0, 8.0])
        value = compute_metrics(y_true, y_pred, ["smape"])["smape"]
        assert np.isfinite(value)
        assert 0.0 < value <= 200.0


class TestRMAE:
    """rMAE: seasonal-naive denominator on the evaluation window."""

    def test_is_one_for_seasonal_naive_forecasts(self):
        """Positive: rMAE == 1 when the model IS the seasonal naive.

        y = [1, 3, 2, 4], k = 2 → naive diffs |2-1|, |4-3| → denominator
        1.0. Predictions are naive for t >= 2; the first k rows carry
        errors summing to 2 so the full-window numerator equals the
        denominator exactly (both are means over different row counts).
        """
        y_true = np.array([1.0, 3.0, 2.0, 4.0])
        y_pred = np.array([2.0, 2.0, 1.0, 3.0])
        value = compute_metrics(y_true, y_pred, ["rmae"], naive_period=2)["rmae"]
        assert value == pytest.approx(1.0)

    def test_denominator_is_same_window(self):
        """Positive: hand-computed rMAE pins the window denominator.

        y = 1..6, pred = ones, k = 2: numerator MAE = 15/6 = 2.5,
        denominator = mean(|3-1|, |4-2|, |5-3|, |6-4|) = 2.0 → 1.25.
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred = np.ones(6)
        value = compute_metrics(y_true, y_pred, ["rmae"], naive_period=2)["rmae"]
        assert value == pytest.approx(1.25)

    def test_requires_naive_period(self):
        """Negative: no naive_period → actionable error, not a crash later."""
        with pytest.raises(ValueError, match="naive_period"):
            compute_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], ["rmae"])

    def test_window_shorter_than_naive_raises(self):
        """Negative (G9): window <= period raises instead of emitting NaN."""
        with pytest.raises(ValueError, match="longer than naive_period"):
            compute_metrics([1.0, 2.0], [1.0, 2.0], ["rmae"], naive_period=2)

    def test_zero_denominator_raises(self):
        """Negative: a constant window makes rMAE undefined (G9)."""
        with pytest.raises(ValueError, match="undefined"):
            compute_metrics(
                [5.0, 5.0, 5.0], [1.0, 2.0, 3.0], ["rmae"], naive_period=1
            )


class TestMASE:
    """MASE: in-sample training denominator, opt-in (G10)."""

    def test_uses_in_sample_denominator(self):
        """Positive: hand-computed MASE with a training-set denominator.

        y_train = 1..10, k = 3 → k-th order diffs all 3 → scale 3.
        Eval MAE = 2 → MASE = 2/3.
        """
        y_train = np.arange(1.0, 11.0)
        y_true = np.array([10.0, 11.0, 12.0])
        y_pred = np.array([9.0, 9.0, 9.0])
        value = compute_metrics(
            y_true, y_pred, ["mase"], y_train=y_train, naive_period=3
        )["mase"]
        assert value == pytest.approx(2.0 / 3.0)

    def test_requires_y_train_and_naive_period(self):
        """Negative: both inputs are mandatory (D2)."""
        y = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="y_train"):
            compute_metrics(y, y, ["mase"], naive_period=1)
        with pytest.raises(ValueError, match="y_train"):
            compute_metrics(y, y, ["mase"], y_train=[1.0, 2.0, 3.0, 4.0])

    def test_train_shorter_than_naive_raises(self):
        """Negative (G9): degenerate in-sample window raises."""
        with pytest.raises(ValueError, match="longer than naive_period"):
            compute_metrics(
                [1.0, 2.0],
                [1.0, 2.0],
                ["mase"],
                y_train=[1.0, 2.0],
                naive_period=2,
            )


class TestRemovedAndUnknownMetrics:
    """D1: removed/unknown names fail loudly, never silently."""

    def test_mape_raises_naming_the_reason(self):
        """Negative: 'mape' raises and the message says why it is gone."""
        with pytest.raises(ValueError, match="mape"):
            compute_metrics([1.0, 2.0], [1.0, 2.0], ["mape"])

    def test_unknown_metric_raises(self):
        """Negative: a typo raises instead of being skipped (old contract)."""
        with pytest.raises(ValueError, match="not_a_metric"):
            compute_metrics([1.0, 2.0], [1.0, 2.0], ["not_a_metric"])

    def test_supported_set_is_the_stage4_vocabulary(self):
        """Contract: the module set matches the Stage-4 metric vocabulary."""
        assert SUPPORTED == {"mae", "rmse", "smape", "rmae", "mase", "r2"}


class TestAdversarialTargets:
    """§0.5 data realities: negative prices and near-zero targets (G8)."""

    def test_all_metrics_finite_on_negatives_and_near_zeros(self):
        """Positive: every Stage-4 metric stays finite on hard targets."""
        y_true = np.array([-20.0, 0.0, 1e-9, 5.0, 100.0])
        y_pred = np.array([-18.0, 0.5, -1e-9, 4.0, 90.0])
        result = compute_metrics(
            y_true,
            y_pred,
            sorted(SUPPORTED),
            y_train=np.arange(1.0, 9.0),
            naive_period=2,
        )
        assert set(result) == set(SUPPORTED)
        assert all(np.isfinite(v) for v in result.values())
