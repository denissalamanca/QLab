"""Phase 3 — Hasbrouck Flow."""

from __future__ import annotations

import numpy as np
import pytest

from afml.features.hasbrouck import hasbrouck_flow


@pytest.mark.phase3
def test_hasbrouck_length_preserved() -> None:
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.standard_normal(500)) + 100.0
    volume = rng.uniform(1.0, 10.0, 500)
    out = hasbrouck_flow(close, volume, window=20)
    assert out.shape == close.shape


@pytest.mark.phase3
def test_hasbrouck_positive_on_monotonic_up() -> None:
    close = 100.0 + np.arange(500) * 0.1
    volume = np.full(500, 4.0)
    out = hasbrouck_flow(close, volume, window=20)
    finite = out[np.isfinite(out)]
    # Steady state: every sign is +1, sqrt(4) = 2, rolling sum over 20 → 40.
    # The first output is 38 (first diff is NaN → sign=0 contributes 0).
    np.testing.assert_allclose(finite[1:], 40.0, atol=1e-9)
    assert 38.0 <= finite[0] <= 40.0


@pytest.mark.phase3
def test_hasbrouck_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        hasbrouck_flow(np.zeros(10), np.zeros(11), window=5)
