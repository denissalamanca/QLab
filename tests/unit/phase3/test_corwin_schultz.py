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


@pytest.mark.phase3
def test_cs_responds_to_bar_durations() -> None:
    """AFML audit V2 — time-adjusted C-S must change with bar_durations.

    On Phase 1 Information Bars (stochastic Δt), the classical C-S formula's
    Brownian-motion assumption breaks. Supplying ``bar_durations`` normalizes
    ``ln²(H/L)`` per unit Δt before the formula is applied, so the output
    differs from the unit-time fallback.
    """
    rng = np.random.default_rng(0)
    n = 500
    mid = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
    high = mid + np.abs(rng.normal(0.01, 0.005, size=n))
    low = mid - np.abs(rng.normal(0.01, 0.005, size=n))

    out_unit = corwin_schultz_spread(high, low, window=2)

    # Stochastic Δt (uniformly between 0.5x and 2x the nominal interval).
    durations = rng.uniform(0.5, 2.0, n)
    out_stoch = corwin_schultz_spread(high, low, bar_durations=durations, window=2)

    a = out_unit[~np.isnan(out_unit)]
    b = out_stoch[~np.isnan(out_stoch)]
    # They must NOT be approximately equal — the time correction is real.
    assert a.size > 0 and b.size > 0
    assert not np.allclose(a[: min(a.size, b.size)], b[: min(a.size, b.size)])


@pytest.mark.phase3
def test_cs_no_durations_matches_unit_durations_explicitly() -> None:
    """``bar_durations=None`` must produce the SAME output as
    ``bar_durations=np.ones(n)`` — the audit-mandated time-adjusted formula
    evaluated at ``Δt = 1``. Differs from the legacy classical formula on
    constant ``Δt ≠ 1`` because the per-unit-time normalization changes scale.
    """
    rng = np.random.default_rng(0)
    n = 200
    mid = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
    high = mid + np.abs(rng.normal(0.01, 0.005, size=n))
    low = mid - np.abs(rng.normal(0.01, 0.005, size=n))

    out_none = corwin_schultz_spread(high, low, window=2)
    out_ones = corwin_schultz_spread(
        high, low, bar_durations=np.ones(n, dtype=np.float64), window=2
    )

    finite = ~np.isnan(out_none) & ~np.isnan(out_ones)
    np.testing.assert_allclose(out_none[finite], out_ones[finite], rtol=1e-12)
