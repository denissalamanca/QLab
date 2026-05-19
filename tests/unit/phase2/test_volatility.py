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


@pytest.mark.phase2
def test_volatility_causal_shift_equals_unshifted_lag() -> None:
    """AFML audit V4 — explicit causality identity.

    ``ewm_volatility(.., causal_shift=True)[t] == ewm_volatility(.., causal_shift=False)[t-1]``
    for every ``t ≥ 1``. The shifted version is precisely the unshifted one
    lagged by one bar.
    """
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(500) * 0.05
    vol_shifted = ewm_volatility(rets, span=20, causal_shift=True)
    vol_unshifted = ewm_volatility(rets, span=20, causal_shift=False)
    # NaN-safe element-wise compare on the lagged region.
    a = vol_shifted[1:]
    b = vol_unshifted[:-1]
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    np.testing.assert_array_equal(a_nan, b_nan)
    np.testing.assert_array_equal(a[~a_nan], b[~b_nan])
    # First element of the shifted series is NaN by construction.
    assert np.isnan(vol_shifted[0])


@pytest.mark.phase2
def test_volatility_does_not_use_return_at_t() -> None:
    """AFML audit V4 — perturbation guard.

    Perturbing ``returns[t]`` must leave ``vol[t]`` unchanged. If it did
    change, future information at index ``t`` would be leaking into the
    barrier-sizing volatility at index ``t``.
    """
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(500) * 0.05
    vol_baseline = ewm_volatility(rets, span=20)

    perturbed = rets.copy()
    perturb_idx = 100
    perturbed[perturb_idx] = 999.0  # massive perturbation at index 100

    vol_perturbed = ewm_volatility(perturbed, span=20)

    # vol at the perturbed index must match — return[100] cannot influence vol[100].
    assert vol_baseline[perturb_idx] == vol_perturbed[perturb_idx] or (
        np.isnan(vol_baseline[perturb_idx]) and np.isnan(vol_perturbed[perturb_idx])
    )
    # vol at strictly earlier indices must also match (return[100] is future for them).
    np.testing.assert_array_equal(
        vol_baseline[: perturb_idx + 1],
        vol_perturbed[: perturb_idx + 1],
        err_msg="vol[t] depends on return[t] — future-data leak",
    )
    # vol from index 101 onward IS allowed to differ (return[100] is now past for them).
    assert not np.array_equal(
        vol_baseline[perturb_idx + 1 :],
        vol_perturbed[perturb_idx + 1 :],
    )
