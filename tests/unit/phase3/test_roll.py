"""Phase 3 — Roll Measure."""

from __future__ import annotations

import numpy as np
import pytest

from afml.features.roll import roll_measure


@pytest.mark.phase3
def test_roll_length_preserved() -> None:
    close = np.cumsum(np.random.default_rng(0).standard_normal(500)) + 100.0
    out = roll_measure(close, window=20)
    assert out.shape == close.shape


@pytest.mark.phase3
def test_roll_warmup_nan() -> None:
    close = np.cumsum(np.random.default_rng(0).standard_normal(500)) + 100.0
    out = roll_measure(close, window=20)
    assert np.all(np.isnan(out[:20]))


@pytest.mark.phase3
def test_roll_zero_on_monotonic_series() -> None:
    """Roll is ≈ 0 when consecutive price changes are perfectly correlated
    (no bid-ask bouncing). Tiny float-cov rounding noise is acceptable."""
    close = 100.0 + np.arange(200) * 0.1
    out = roll_measure(close, window=30)
    finite = out[np.isfinite(out)]
    np.testing.assert_allclose(finite, 0.0, atol=1e-6)


@pytest.mark.phase3
def test_roll_positive_on_bid_ask_bouncing() -> None:
    """A perfect bid-ask bounce around a trend produces negative
    Cov(ΔP_t, ΔP_{t-1}) → Roll > 0."""
    rng = np.random.default_rng(0)
    n = 500
    mid = 100.0 + np.cumsum(rng.standard_normal(n) * 0.01)
    # Add tick-level bid/ask bounce at every other bar.
    close = mid + np.tile([0.05, -0.05], n // 2 + 1)[:n]
    out = roll_measure(close, window=30)
    finite = out[np.isfinite(out)]
    assert np.mean(finite > 0.0) > 0.8


@pytest.mark.phase3
def test_roll_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        roll_measure(np.zeros((10, 2)), window=5)
    with pytest.raises(ValueError, match="window"):
        roll_measure(np.zeros(100), window=1)
