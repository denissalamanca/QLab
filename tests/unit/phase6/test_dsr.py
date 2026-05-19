"""Phase 6 — Deflated Sharpe Ratio + Expected Max Sharpe."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from afml.validation.dsr import (
    EULER_MASCHERONI,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)


@pytest.mark.phase6
def test_expected_max_sharpe_grows_with_n_trials() -> None:
    """E[max{SR}] is monotonically increasing in N."""
    e_10 = expected_max_sharpe(10, 1.0)
    e_100 = expected_max_sharpe(100, 1.0)
    e_1000 = expected_max_sharpe(1000, 1.0)
    assert e_10 < e_100 < e_1000


@pytest.mark.phase6
def test_expected_max_sharpe_scales_linearly_with_std() -> None:
    """E[max{SR}] is linear in σ_SR (Bailey-Lopez de Prado eqn.)."""
    e_unit = expected_max_sharpe(50, 1.0)
    e_scaled = expected_max_sharpe(50, 2.0)
    assert np.isclose(e_scaled, e_unit * 2.0, rtol=1e-10)


@pytest.mark.phase6
def test_expected_max_sharpe_formula_matches_reference() -> None:
    """Compute the formula directly and check against the helper."""
    n = 50
    z_a = norm.ppf(1.0 - 1.0 / n)
    z_b = norm.ppf(1.0 - 1.0 / (n * np.e))
    expected = z_a * (1.0 - EULER_MASCHERONI) + z_b * EULER_MASCHERONI
    assert np.isclose(expected_max_sharpe(n, 1.0), expected, rtol=1e-12)


@pytest.mark.phase6
def test_dsr_low_on_pure_noise() -> None:
    """A noise return series after multiple-testing deflation should have
    DSR << 0.5."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, 252)  # zero-mean returns
    result = deflated_sharpe_ratio(returns, n_trials=100)
    assert result.dsr < 0.5, f"DSR on noise should be < 0.5, got {result.dsr:.3f}"


@pytest.mark.phase6
def test_dsr_higher_on_genuine_signal() -> None:
    """A return series with a strong positive mean must have higher DSR than
    one with zero mean, all else equal."""
    rng = np.random.default_rng(0)
    n_obs = 252
    noise = rng.normal(0.0, 0.01, n_obs)
    signal = rng.normal(0.002, 0.01, n_obs)  # ~3.2 annualized Sharpe
    dsr_noise = deflated_sharpe_ratio(noise, n_trials=10).dsr
    dsr_signal = deflated_sharpe_ratio(signal, n_trials=10).dsr
    assert dsr_signal > dsr_noise


@pytest.mark.phase6
def test_dsr_returns_observed_sharpe_annualized() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, 252)
    result = deflated_sharpe_ratio(returns, n_trials=10, periods_per_year=252)
    sharpe_period = float(np.mean(returns)) / float(np.std(returns, ddof=1))
    expected_annualized = sharpe_period * np.sqrt(252)
    assert np.isclose(result.sharpe_observed, expected_annualized, rtol=1e-9)


@pytest.mark.phase6
def test_dsr_rejects_bad_input() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="1-D"):
        deflated_sharpe_ratio(rng.standard_normal((10, 2)), n_trials=5)
    with pytest.raises(ValueError, match="finite"):
        bad = np.array([1.0, np.nan, 3.0])
        deflated_sharpe_ratio(bad, n_trials=5)
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_ratio(rng.standard_normal(20), n_trials=0)


@pytest.mark.phase6
def test_dsr_degenerate_returns_half() -> None:
    """Zero-variance returns ⇒ DSR = 0.5 (no signal either way)."""
    returns = np.full(20, 0.001)
    result = deflated_sharpe_ratio(returns, n_trials=5)
    assert result.dsr == 0.5
