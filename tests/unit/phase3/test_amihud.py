"""Phase 3 — Amihud's Lambda."""

from __future__ import annotations

import numpy as np
import pytest

from afml.features.amihud import amihud_lambda


@pytest.mark.phase3
def test_amihud_length_preserved() -> None:
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.standard_normal(500)) + 100.0
    volume = rng.uniform(1.0, 10.0, 500)
    out = amihud_lambda(close, volume, window=20)
    assert out.shape == close.shape


@pytest.mark.phase3
def test_amihud_finite_on_typical_data() -> None:
    """AFML audit V1 — rolling-OLS Amihud yields finite slopes on typical data.

    Note: as a regression slope rather than an element-wise mean, Amihud is
    NO LONGER constrained to be ≥ 0 — but it must remain finite (no inf/NaN
    outliers caused by near-zero volume in the denominator).
    """
    rng = np.random.default_rng(0)
    close = np.abs(np.cumsum(rng.standard_normal(500)) + 100.0) + 1.0
    volume = rng.uniform(1.0, 10.0, 500)
    out = amihud_lambda(close, volume, window=20)
    finite = out[np.isfinite(out)]
    assert finite.size > 0
    assert not np.any(np.isinf(out)), "Amihud OLS must never produce inf"


@pytest.mark.phase3
def test_amihud_zero_on_flat_prices() -> None:
    """Zero returns → zero |return| → OLS slope of zeros on dollar_volume = 0."""
    close = np.full(500, 100.0)
    volume = np.full(500, 5.0)
    out = amihud_lambda(close, volume, window=20)
    finite = out[np.isfinite(out)]
    np.testing.assert_allclose(finite, 0.0, atol=1e-15)


@pytest.mark.phase3
def test_amihud_no_inf_with_near_zero_volume() -> None:
    """AFML audit V1 — rolling OLS must NOT produce inf when volume is tiny.

    The element-wise ``|return| / $volume`` Amihud would blow up here; the
    rolling-OLS replacement gracefully handles small variance via the
    ``MIN_VARIANCE`` guard.
    """
    rng = np.random.default_rng(0)
    n = 500
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
    volume = np.full(n, 1e-14)  # near zero
    out = amihud_lambda(close, volume, window=20)
    assert not np.any(np.isinf(out)), "Amihud OLS must never produce inf"


@pytest.mark.phase3
def test_amihud_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        amihud_lambda(np.zeros(10), np.zeros(11), window=5)
