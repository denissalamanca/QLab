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
def test_shannon_bits_in_range_0_to_log2_nbins() -> None:
    """AFML audit V4 — quintile (default ``n_bins = 5``) alphabet ⇒ max entropy
    is ``log₂(5) ≈ 2.32`` bits."""
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.standard_normal(1000) * 0.05) + 100.0
    out = shannon_entropy(close, window=100)
    finite = out[np.isfinite(out)]
    assert np.all(finite >= 0.0)
    assert np.all(finite <= np.log2(5) + 1e-6)


@pytest.mark.phase3
def test_shannon_distinguishes_volatility_states() -> None:
    """AFML audit V4 — quantile discretization captures volatility, not just
    direction.

    Two series with the SAME up/down sign pattern but DIFFERENT magnitude
    distributions must produce different entropies. Under the old binary-sign
    encoding both would have identical entropy.
    """
    rng = np.random.default_rng(0)
    n = 600
    # Series A: pure 50/50 alternation with constant magnitude (low vol-state).
    rets_a = np.tile([0.001, -0.001], n // 2)
    close_a = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], rets_a])))

    # Series B: same sign frequency overall, but a MIX of magnitudes —
    # 4 distinct return values spanning low- and high-vol bars.
    half = n // 2
    pos = rng.choice([0.0005, 0.001, 0.002, 0.005], size=half)
    neg = -rng.choice([0.0005, 0.001, 0.002, 0.005], size=half)
    rets_b = np.empty(n)
    rets_b[0::2] = pos
    rets_b[1::2] = neg
    close_b = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], rets_b])))

    out_a = shannon_entropy(close_a, window=50)
    out_b = shannon_entropy(close_b, window=50)

    mean_a = float(np.nanmean(out_a))
    mean_b = float(np.nanmean(out_b))
    # B's varied magnitudes populate more quantile bins → higher entropy.
    assert mean_b > mean_a + 0.1, (
        f"quantile encoder must distinguish volatility states: H_A={mean_a:.3f}, H_B={mean_b:.3f}"
    )
