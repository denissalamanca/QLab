"""Phase 2 — EWM volatility wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from afml.labeling.volatility import ewm_volatility


@pytest.mark.phase2
def test_volatility_returns_same_length() -> None:
    rets = np.random.default_rng(0).standard_normal(500)
    out = ewm_volatility(rets, span=50)
    assert out.shape == rets.shape


@pytest.mark.phase2
def test_volatility_causal_shift_inserts_nan_prefix() -> None:
    """With ``causal_shift=True``, the EWM at index ``t`` uses returns strictly
    before ``t`` → the first ``span`` indices are NaN."""
    rets = np.random.default_rng(0).standard_normal(500)
    out = ewm_volatility(rets, span=50, causal_shift=True)
    # The first 50 entries are NaN warm-up; index 50 = first finite value.
    assert np.all(np.isnan(out[:50]))
    assert np.isfinite(out[51:]).all()


@pytest.mark.phase2
def test_volatility_no_causal_shift_finite_from_span_minus_1() -> None:
    rets = np.random.default_rng(0).standard_normal(500)
    out = ewm_volatility(rets, span=50, causal_shift=False)
    assert np.all(np.isnan(out[:49]))
    assert np.isfinite(out[49:]).all()


@pytest.mark.phase2
def test_volatility_constant_returns_yield_zero() -> None:
    out = ewm_volatility(np.full(500, 0.01), span=20)
    finite = out[~np.isnan(out)]
    assert finite.size > 0
    np.testing.assert_allclose(finite, 0.0, atol=1e-12)


@pytest.mark.phase2
def test_volatility_gaussian_returns_recovers_sigma() -> None:
    """For a long iid Gaussian sample, EWM std converges to the population σ."""
    rng = np.random.default_rng(0)
    sigma = 0.05
    rets = rng.normal(0, sigma, size=20_000)
    out = ewm_volatility(rets, span=200)
    finite = out[~np.isnan(out)]
    # EWM std is biased downward for finite span; allow generous tolerance.
    assert 0.7 * sigma < float(np.mean(finite[-5000:])) < 1.3 * sigma


@pytest.mark.phase2
def test_volatility_raises_on_2d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        ewm_volatility(np.zeros((10, 2)), span=5)
