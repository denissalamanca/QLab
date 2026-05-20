"""M1.5 — plateau objective: median PWF OOS strategy Sharpe."""

from __future__ import annotations

import numpy as np
import pytest

from afml.execution import calculate_bet_size
from afml.modeling import BrainTwoResult, FoldOOS
from afml.research.objective import oos_strategy_sharpe

pytestmark = pytest.mark.m1


def _result(folds: list[FoldOOS] | None, n: int) -> BrainTwoResult:
    return BrainTwoResult(
        fold_diagnostics=[],
        average_uniqueness_per_event=np.zeros(n, dtype=np.float64),
        last_calibration=None,
        oos_predictions=folds,
    )


def test_single_fold_matches_manual_sharpe() -> None:
    idx = np.array([0, 1, 2, 3], dtype=np.int64)
    probs = np.full(4, 0.75, dtype=np.float64)
    ret = np.array([0.01, -0.01, 0.02, -0.005], dtype=np.float64)
    side = np.ones(4, dtype=np.int64)

    got = oos_strategy_sharpe(
        _result([FoldOOS(0, idx, probs)], 4), ret, side, periods_per_year=252.0
    )

    bet = calculate_bet_size(0.75)
    strat = bet * ret
    expected = float(strat.mean() / strat.std(ddof=1) * np.sqrt(252.0))
    assert got == pytest.approx(expected)


def test_short_side_profits_on_down_moves() -> None:
    idx = np.array([0, 1, 2, 3], dtype=np.int64)
    probs = np.full(4, 0.8, dtype=np.float64)
    ret = np.array([-0.01, -0.02, -0.015, -0.03], dtype=np.float64)  # market falls
    side = -np.ones(4, dtype=np.int64)  # short
    got = oos_strategy_sharpe(
        _result([FoldOOS(0, idx, probs)], 4), ret, side, periods_per_year=252.0
    )
    assert got is not None and got > 0.0  # shorts make money on down moves


def test_median_across_folds() -> None:
    fold_a = FoldOOS(0, np.array([0, 1, 2], dtype=np.int64), np.full(3, 0.8, dtype=np.float64))
    fold_b = FoldOOS(1, np.array([3, 4, 5], dtype=np.int64), np.full(3, 0.7, dtype=np.float64))
    ret = np.array([0.02, 0.01, 0.03, -0.02, 0.01, -0.01], dtype=np.float64)
    side = np.ones(6, dtype=np.int64)

    res = _result([fold_a, fold_b], 6)
    got = oos_strategy_sharpe(res, ret, side, periods_per_year=252.0)

    # Recompute each fold independently; median of two = mean.
    def _fold_sharpe(f: FoldOOS) -> float:
        bets = np.array([calculate_bet_size(float(p)) for p in f.calibrated_proba])
        strat = bets * ret[f.holdout_indices]
        return float(strat.mean() / strat.std(ddof=1) * np.sqrt(252.0))

    expected = float(np.median([_fold_sharpe(fold_a), _fold_sharpe(fold_b)]))
    assert got == pytest.approx(expected)


def test_zero_variance_fold_returns_none() -> None:
    idx = np.array([0, 1], dtype=np.int64)
    probs = np.full(2, 0.7, dtype=np.float64)
    ret = np.array([0.01, 0.01], dtype=np.float64)  # constant → zero-variance strat
    side = np.ones(2, dtype=np.int64)
    assert (
        oos_strategy_sharpe(_result([FoldOOS(0, idx, probs)], 2), ret, side, periods_per_year=252.0)
        is None
    )


def test_missing_oos_predictions_raises() -> None:
    ret = np.array([0.01, -0.01], dtype=np.float64)
    side = np.ones(2, dtype=np.int64)
    with pytest.raises(ValueError, match="oos_predictions"):
        oos_strategy_sharpe(_result(None, 2), ret, side, periods_per_year=252.0)
