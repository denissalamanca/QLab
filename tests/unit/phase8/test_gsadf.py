"""Phase 8 — GSADF explosive-root bubble detection (Blueprint §10.1/§10.3)."""

from __future__ import annotations

import numpy as np
import pytest

from afml.monitoring.gsadf import (
    detect_bubble,
    gsadf_critical_value,
    gsadf_statistic,
)


def _random_walk(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) + 100.0


def _exponential_bubble(n: int, seed: int, *, onset: int, rate: float = 1.04) -> np.ndarray:
    """Random walk that turns mildly explosive at ``onset``."""
    series = _random_walk(n, seed)
    series[onset:] = series[onset] * (rate ** np.arange(n - onset))
    return series


def _mean_reverting(n: int, seed: int, *, phi: float = 0.3) -> np.ndarray:
    """Strongly mean-reverting AR(1) — the opposite of explosive."""
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = 0.0
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y + 100.0


@pytest.mark.phase8
def test_bubble_detection_triggers_on_exponential_data() -> None:
    """Blueprint §10.3 DoD — synthetic exponential data ⇒ alert triggered."""
    series = _exponential_bubble(120, seed=0, onset=80)
    result = detect_bubble(series, n_simulations=99, random_state=1)
    assert result.is_bubble is True
    assert result.gsadf_statistic > result.critical_value


@pytest.mark.phase8
def test_mean_reverting_series_does_not_trigger() -> None:
    """A strongly mean-reverting series is the antithesis of explosive — its
    GSADF must sit below the random-walk critical value (no false alarm)."""
    series = _mean_reverting(150, seed=0)
    result = detect_bubble(series, n_simulations=199, random_state=2)
    assert result.is_bubble is False


@pytest.mark.phase8
def test_bubble_statistic_dominates_random_walk() -> None:
    """The explosive series' GSADF must vastly exceed a random walk's —
    a seed-robust sanity check on the statistic's discriminating power."""
    bubble = _exponential_bubble(120, seed=3, onset=80)
    rw = _random_walk(120, seed=3)
    assert gsadf_statistic(bubble) > gsadf_statistic(rw)


@pytest.mark.phase8
def test_critical_value_monotone_in_quantile() -> None:
    """Higher confidence ⇒ higher critical value."""
    crit_90 = gsadf_critical_value(80, n_simulations=99, quantile=0.90, random_state=0)
    crit_99 = gsadf_critical_value(80, n_simulations=99, quantile=0.99, random_state=0)
    assert crit_99 > crit_90


@pytest.mark.phase8
def test_gsadf_reproducible_with_seed() -> None:
    series = _exponential_bubble(100, seed=5, onset=70)
    r1 = detect_bubble(series, n_simulations=50, random_state=7)
    r2 = detect_bubble(series, n_simulations=50, random_state=7)
    assert r1.gsadf_statistic == r2.gsadf_statistic
    assert r1.critical_value == r2.critical_value


@pytest.mark.phase8
def test_gsadf_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        gsadf_statistic(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="finite"):
        bad = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        gsadf_statistic(bad)


@pytest.mark.phase8
def test_gsadf_statistic_independent_of_level_shift() -> None:
    """Adding a constant to the whole series shifts the level but not the
    autoregressive dynamics — GSADF (intercept-included) is invariant."""
    series = _random_walk(100, seed=11)
    stat_a = gsadf_statistic(series)
    stat_b = gsadf_statistic(series + 500.0)
    assert stat_a == pytest.approx(stat_b, rel=1e-6)
