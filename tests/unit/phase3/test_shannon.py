"""Phase 3 — Shannon Entropy."""

from __future__ import annotations

import numpy as np
import pytest

from afml.features.shannon import shannon_entropy


@pytest.mark.phase3
def test_shannon_length_preserved() -> None:
    close = np.cumsum(np.random.default_rng(0).standard_normal(500)) + 100.0
    out = shannon_entropy(close, window=50)
    assert out.shape == close.shape


@pytest.mark.phase3
def test_shannon_max_on_balanced_signs() -> None:
    """A perfect 50/50 split between +1 and -1 signs gives H = 1 bit."""
    close = 100.0 + np.tile([0.05, 0.0], 250)
    out = shannon_entropy(close, window=20)
    finite = out[np.isfinite(out)]
    np.testing.assert_allclose(finite[-50:], 1.0, atol=0.1)


@pytest.mark.phase3
def test_shannon_min_on_monotonic() -> None:
    """All bars trending up → all +1 signs → H = 0."""
    close = 100.0 + np.arange(500) * 0.1
    out = shannon_entropy(close, window=20)
    finite = out[np.isfinite(out)]
    np.testing.assert_allclose(finite[-100:], 0.0, atol=1e-9)


@pytest.mark.phase3
def test_shannon_bits_in_range_0_to_log2_3() -> None:
    """Trinary alphabet → max entropy is log2(3) ≈ 1.585 bits."""
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.standard_normal(1000) * 0.05) + 100.0
    out = shannon_entropy(close, window=100)
    finite = out[np.isfinite(out)]
    assert np.all(finite >= 0.0)
    assert np.all(finite <= np.log2(3) + 1e-6)
