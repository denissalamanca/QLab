"""Phase 7 — probabilistic bet sizing (Blueprint §9.1)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from afml.execution.bet_sizing import (
    bet_size_from_probability,
    bet_sizes_for_batch,
    calculate_bet_size,
)


@pytest.mark.phase7
def test_zero_size_below_half() -> None:
    """Blueprint §9.3 — ``calculate_bet_size(p=0.49) == 0.0``."""
    assert calculate_bet_size(0.49) == 0.0


@pytest.mark.phase7
def test_zero_size_at_exactly_half() -> None:
    assert calculate_bet_size(0.5) == 0.0


@pytest.mark.phase7
def test_size_increases_monotonically_with_probability() -> None:
    probs = [0.51, 0.6, 0.7, 0.8, 0.9, 0.99]
    sizes = [calculate_bet_size(p) for p in probs]
    for a, b in pairwise(sizes):
        assert b > a, f"bet size not monotone: {sizes}"


@pytest.mark.phase7
def test_size_bounded_zero_one() -> None:
    for p in np.linspace(0.0, 1.0, 101):
        size = calculate_bet_size(float(p))
        assert 0.0 <= size <= 1.0


@pytest.mark.phase7
def test_size_approaches_one_at_high_confidence() -> None:
    assert calculate_bet_size(0.999) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.phase7
def test_known_value_at_p_0_6() -> None:
    """z = (0.6-0.5)/sqrt(0.6*0.4) = 0.2041; size = 2Φ(z)-1 ≈ 0.1617."""
    assert calculate_bet_size(0.6) == pytest.approx(0.1617, abs=1e-3)


@pytest.mark.phase7
def test_bet_size_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="in"):
        bet_size_from_probability(1.5)
    with pytest.raises(ValueError, match="in"):
        bet_size_from_probability(-0.1)
    with pytest.raises(ValueError, match="finite"):
        bet_size_from_probability(float("nan"))


@pytest.mark.phase7
def test_batch_zeros_below_half() -> None:
    probs = np.array([0.3, 0.49, 0.5, 0.7, 0.9])
    out = bet_sizes_for_batch(probs)
    assert out.sizes[0] == 0.0
    assert out.sizes[1] == 0.0
    assert out.sizes[2] == 0.0
    assert out.sizes[3] > 0.0
    assert out.sizes[4] > out.sizes[3]


@pytest.mark.phase7
def test_batch_gaussian_does_not_trigger_fallback() -> None:
    """A unimodal, near-Gaussian batch of probabilities should NOT trip the
    Mixture-of-Gaussians fallback."""
    rng = np.random.default_rng(0)
    # Probabilities clustered around 0.7 → unimodal z-distribution.
    probs = np.clip(rng.normal(0.7, 0.05, 200), 0.51, 0.999)
    out = bet_sizes_for_batch(probs)
    assert not out.used_mixture_fallback


@pytest.mark.phase7
def test_batch_bimodal_triggers_mixture_fallback() -> None:
    """A clearly bimodal probability batch (two regimes) should fail the
    Shapiro-Wilk normality test and engage the mixture fallback."""
    rng = np.random.default_rng(0)
    low = rng.normal(0.55, 0.01, 150)  # barely-confident cluster
    high = rng.normal(0.97, 0.005, 150)  # very-confident cluster
    probs = np.clip(np.concatenate([low, high]), 0.51, 0.999)
    out = bet_sizes_for_batch(probs)
    assert out.used_mixture_fallback
    assert out.shapiro_pvalue < 0.05
    # Sizes still bounded and positive.
    assert np.all(out.sizes >= 0.0)
    assert np.all(out.sizes <= 1.0)


@pytest.mark.phase7
def test_batch_all_below_half_returns_zeros() -> None:
    probs = np.array([0.1, 0.3, 0.45, 0.5])
    out = bet_sizes_for_batch(probs)
    assert np.all(out.sizes == 0.0)
    assert not out.used_mixture_fallback
