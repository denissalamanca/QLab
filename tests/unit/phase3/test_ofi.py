"""Phase 3 — Order Flow Imbalance."""

from __future__ import annotations

import numpy as np
import pytest

from afml.features.ofi import ofi


@pytest.mark.phase3
def test_ofi_length_preserved() -> None:
    close = np.cumsum(np.random.default_rng(0).standard_normal(500)) + 100.0
    volume = np.full(500, 1.0)
    out = ofi(close, volume, window=20)
    assert out.shape == close.shape


@pytest.mark.phase3
def test_ofi_positive_on_monotonic_up() -> None:
    close = 100.0 + np.arange(500) * 0.1
    volume = np.full(500, 1.0)
    out = ofi(close, volume, window=20)
    finite = out[np.isfinite(out)]
    # Every bar's sign is +1, volume = 1, window = 20.
    # The very first finite output incorporates the first-diff NaN (treated as
    # sign 0), so the rolling sum is 19 at that single index. Steady state is
    # 20.0 for all subsequent outputs.
    np.testing.assert_allclose(finite[1:], 20.0, atol=1e-9)
    assert 19.0 <= finite[0] <= 20.0


@pytest.mark.phase3
def test_ofi_zero_on_perfect_alternation() -> None:
    close = 100.0 + np.tile([0.05, 0.0], 250)
    volume = np.full(500, 1.0)
    out = ofi(close, volume, window=20)
    finite = out[np.isfinite(out)]
    # Alternating +/- signs with equal volume → rolling sum ≈ 0.
    assert np.all(np.abs(finite) <= 1.0)


@pytest.mark.phase3
def test_ofi_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        ofi(np.zeros(10), np.zeros(11), window=5)
