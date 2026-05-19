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
def test_amihud_nonnegative() -> None:
    """|return| / dollar_volume is always ≥ 0."""
    rng = np.random.default_rng(0)
    close = np.abs(np.cumsum(rng.standard_normal(500)) + 100.0) + 1.0
    volume = rng.uniform(1.0, 10.0, 500)
    out = amihud_lambda(close, volume, window=20)
    finite = out[np.isfinite(out)]
    assert np.all(finite >= 0.0)


@pytest.mark.phase3
def test_amihud_zero_on_flat_prices() -> None:
    """Zero returns → zero Amihud."""
    close = np.full(500, 100.0)
    volume = np.full(500, 5.0)
    out = amihud_lambda(close, volume, window=20)
    finite = out[np.isfinite(out)]
    np.testing.assert_allclose(finite, 0.0, atol=1e-15)


@pytest.mark.phase3
def test_amihud_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        amihud_lambda(np.zeros(10), np.zeros(11), window=5)
