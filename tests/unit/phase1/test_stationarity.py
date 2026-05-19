"""Phase 1 — ADF and Jarque-Bera wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from afml.data.stationarity import adf_pvalue, jarque_bera_statistic


@pytest.mark.phase1
def test_adf_rejects_random_walk() -> None:
    """Random walk (∼I(1)) should fail ADF: p > 0.05."""
    rng = np.random.default_rng(42)
    rw = np.cumsum(rng.standard_normal(2000))
    assert adf_pvalue(rw) > 0.05


@pytest.mark.phase1
def test_adf_passes_stationary_ar1() -> None:
    """AR(1) with |coef|<1 is stationary: p < 0.05."""
    rng = np.random.default_rng(7)
    n = 2000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.3 * x[t - 1] + rng.standard_normal()
    assert adf_pvalue(x) < 0.05


@pytest.mark.phase1
def test_adf_ignores_nan_prefix() -> None:
    rng = np.random.default_rng(11)
    n = 1500
    series = np.empty(n + 100)
    series[:100] = np.nan
    series[100:] = rng.standard_normal(n) * 0.1
    # tail is i.i.d. Gaussian → stationary
    assert adf_pvalue(series) < 0.05


@pytest.mark.phase1
def test_adf_raises_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        adf_pvalue(np.array([1.0, 2.0, 3.0]))


@pytest.mark.phase1
def test_jb_low_for_gaussian() -> None:
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(5000)
    assert jarque_bera_statistic(samples) < 50.0


@pytest.mark.phase1
def test_jb_high_for_heavy_tailed() -> None:
    rng = np.random.default_rng(0)
    samples = rng.standard_t(df=3, size=5000)  # heavy tails
    assert jarque_bera_statistic(samples) > 200.0


@pytest.mark.phase1
def test_jb_raises_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        jarque_bera_statistic(np.array([1.0, 2.0]))
