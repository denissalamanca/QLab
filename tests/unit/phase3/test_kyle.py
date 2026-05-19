"""Phase 3 — Kyle's Lambda."""

from __future__ import annotations

import numpy as np
import pytest

from afml.features.kyle import kyle_lambda


@pytest.mark.phase3
def test_kyle_length_preserved() -> None:
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.standard_normal(500)) + 100.0
    volume = rng.uniform(1.0, 10.0, 500)
    out = kyle_lambda(close, volume, window=20)
    assert out.shape == close.shape


@pytest.mark.phase3
def test_kyle_finite_values_on_random_data() -> None:
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.standard_normal(1000) * 0.1) + 100.0
    volume = rng.uniform(1.0, 10.0, 1000)
    out = kyle_lambda(close, volume, window=50)
    finite = out[np.isfinite(out)]
    assert finite.size > 0
    assert np.all(np.isfinite(finite))


@pytest.mark.phase3
def test_kyle_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        kyle_lambda(np.zeros(10), np.zeros(11), window=5)


@pytest.mark.phase3
def test_kyle_warmup_nan() -> None:
    close = np.cumsum(np.random.default_rng(0).standard_normal(200)) + 100.0
    volume = np.full(200, 1.0)
    out = kyle_lambda(close, volume, window=30)
    assert np.all(np.isnan(out[:30]))
