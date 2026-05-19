"""Phase 3 — Corwin-Schultz Bid-Ask Spread."""

from __future__ import annotations

import numpy as np
import pytest

from afml.features.corwin_schultz import corwin_schultz_spread


@pytest.mark.phase3
def test_cs_length_preserved() -> None:
    rng = np.random.default_rng(0)
    n = 500
    mid = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
    high = mid + np.abs(rng.normal(0.01, 0.005, size=n))
    low = mid - np.abs(rng.normal(0.01, 0.005, size=n))
    out = corwin_schultz_spread(high, low, window=2)
    assert out.shape == high.shape


@pytest.mark.phase3
def test_cs_nonnegative_clip() -> None:
    """Negative α paths must clip to 0, never produce a negative spread."""
    rng = np.random.default_rng(0)
    n = 500
    mid = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
    high = mid + np.abs(rng.normal(0.01, 0.005, size=n))
    low = mid - np.abs(rng.normal(0.01, 0.005, size=n))
    out = corwin_schultz_spread(high, low, window=5)
    finite = out[np.isfinite(out)]
    assert np.all(finite >= 0.0)


@pytest.mark.phase3
def test_cs_zero_when_high_equals_low() -> None:
    """If high == low (no intraday range) the formula yields spread = 0."""
    n = 100
    same = np.full(n, 100.0)
    out = corwin_schultz_spread(same, same, window=2)
    finite = out[np.isfinite(out)]
    np.testing.assert_allclose(finite, 0.0, atol=1e-9)


@pytest.mark.phase3
def test_cs_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        corwin_schultz_spread(np.zeros(10), np.zeros(11), window=2)


@pytest.mark.phase3
def test_cs_rejects_window_too_small() -> None:
    with pytest.raises(ValueError, match="window"):
        corwin_schultz_spread(np.zeros(10), np.zeros(10), window=1)
