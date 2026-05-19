"""Phase 6 — PBO (Probability of Backtest Overfitting)."""

from __future__ import annotations

import numpy as np
import pytest

from afml.validation.pbo import compute_pbo


@pytest.mark.phase6
def test_pbo_on_pure_noise_is_near_half() -> None:
    """When IS and OOS performances are independent noise, the IS-best
    strategy's OOS rank is uniform over [0, 1]; PBO concentrates near 0.5."""
    rng = np.random.default_rng(0)
    n_splits = 200
    n_strategies = 20
    is_perf = rng.standard_normal((n_splits, n_strategies))
    oos_perf = rng.standard_normal((n_splits, n_strategies))
    result = compute_pbo(is_perf, oos_perf)
    assert 0.35 < result.pbo < 0.65, f"PBO on noise should be ~0.5, got {result.pbo:.3f}"


@pytest.mark.phase6
def test_pbo_low_when_is_predicts_oos() -> None:
    """When the IS-best strategy is reliably the OOS-best too (correlated
    IS / OOS perf), PBO drops well below 0.5."""
    rng = np.random.default_rng(0)
    n_splits = 100
    n_strategies = 20
    # Each strategy has a true mean; we see noisy estimates IS and OOS.
    true_mean = rng.standard_normal(n_strategies) * 0.5
    noise_is = rng.standard_normal((n_splits, n_strategies)) * 0.1
    noise_oos = rng.standard_normal((n_splits, n_strategies)) * 0.1
    is_perf = true_mean + noise_is
    oos_perf = true_mean + noise_oos
    result = compute_pbo(is_perf, oos_perf)
    assert result.pbo < 0.1, f"PBO should be tiny when IS reliably ranks OOS, got {result.pbo:.3f}"


@pytest.mark.phase6
def test_pbo_high_under_anti_skill() -> None:
    """If IS-best deliberately picks OOS-worst (anti-skill), PBO → 1."""
    rng = np.random.default_rng(0)
    n_splits = 100
    n_strategies = 20
    # OOS is the negation of IS — IS-best becomes OOS-worst.
    is_perf = rng.standard_normal((n_splits, n_strategies))
    oos_perf = -is_perf
    result = compute_pbo(is_perf, oos_perf)
    assert result.pbo > 0.9, f"PBO under perfect anti-skill should be near 1, got {result.pbo:.3f}"


@pytest.mark.phase6
def test_pbo_returns_per_split_logits() -> None:
    rng = np.random.default_rng(0)
    is_perf = rng.standard_normal((15, 10))
    oos_perf = rng.standard_normal((15, 10))
    result = compute_pbo(is_perf, oos_perf)
    assert result.logits.shape == (15,)
    assert result.n_splits == 15
    assert result.n_strategies == 10


@pytest.mark.phase6
def test_pbo_rejects_bad_input() -> None:
    rng = np.random.default_rng(0)
    is_perf = rng.standard_normal((10, 5))
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_pbo(is_perf, rng.standard_normal((10, 6)))
    with pytest.raises(ValueError, match="finite"):
        bad = is_perf.copy()
        bad[0, 0] = np.nan
        compute_pbo(bad, is_perf)


@pytest.mark.phase6
def test_pbo_rejects_single_strategy_matrix() -> None:
    """AFML Phase 0-6 audit V1 — PBO on a single strategy (n×1 matrix) is
    mathematically meaningless and must raise with a cohort-required message."""
    rng = np.random.default_rng(0)
    single = rng.standard_normal((15, 1))
    with pytest.raises(ValueError, match=r"single strategy|cohort"):
        compute_pbo(single, single)
