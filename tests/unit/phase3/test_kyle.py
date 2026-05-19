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


@pytest.mark.phase3
def test_kyle_no_inf_with_near_zero_volume() -> None:
    """AFML audit V1 — rolling OLS guards against near-zero signed-volume
    variance and emits NaN rather than ±inf.
    """
    rng = np.random.default_rng(0)
    n = 300
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
    volume = np.full(n, 1e-14)
    out = kyle_lambda(close, volume, window=20)
    assert not np.any(np.isinf(out)), "Kyle's λ OLS must never produce inf"


@pytest.mark.phase3
def test_kyle_recovers_planted_slope() -> None:
    """AFML audit V1 — when ΔP is deliberately linear in signed volume with a
    known slope, the rolling OLS estimator should recover it within tolerance.
    """
    rng = np.random.default_rng(0)
    n = 600
    true_lambda = 0.002
    # Signed flow drives the bar's price change linearly + small noise.
    signed_flow = rng.standard_normal(n) * 100.0
    dp = true_lambda * signed_flow + rng.standard_normal(n) * 0.01
    close = 100.0 + np.cumsum(dp)
    # Reconstruct the (unsigned) volume that would produce this signed flow.
    volume = np.abs(signed_flow)

    out = kyle_lambda(close, volume, window=100)
    finite = out[np.isfinite(out)]
    assert finite.size > 50
    # The mean of the rolling estimate should converge close to true_lambda.
    assert abs(float(np.mean(finite)) - true_lambda) < 5e-4
