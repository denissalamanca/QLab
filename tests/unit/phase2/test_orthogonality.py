"""Phase 2 — Orthogonality check against deployed signals."""

from __future__ import annotations

import numpy as np
import pytest

from afml.labeling.orthogonality import is_orthogonal, max_correlation


@pytest.mark.phase2
def test_empty_existing_returns_zero() -> None:
    sig = np.array([1.0, 0.0, 1.0, 0.0])
    assert max_correlation(sig, []) == 0.0
    assert is_orthogonal(sig, [])


@pytest.mark.phase2
def test_identical_signal_yields_unit_correlation() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(500)
    assert max_correlation(a, [a]) == pytest.approx(1.0)
    assert not is_orthogonal(a, [a])


@pytest.mark.phase2
def test_perfectly_orthogonal_signals_yield_zero_correlation() -> None:
    """Sine and cosine on the same grid are orthogonal."""
    t = np.linspace(0, 4 * np.pi, 1000)
    sine = np.sin(t)
    cosine = np.cos(t)
    assert max_correlation(sine, [cosine]) < 0.05
    assert is_orthogonal(sine, [cosine], threshold=0.05)


@pytest.mark.phase2
def test_negative_correlation_uses_absolute_value() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(500)
    assert max_correlation(a, [-a]) == pytest.approx(1.0)


@pytest.mark.phase2
def test_unequal_lengths_truncate_to_shorter() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(500)
    b = a[:300]
    assert max_correlation(a, [b]) == pytest.approx(1.0)


@pytest.mark.phase2
def test_zero_variance_signals_skipped() -> None:
    sig = np.array([1.0, 2.0, 3.0, 4.0])
    flat = np.full(4, 5.0)
    # std(flat) = 0 → skip; no usable existing → 0.0
    assert max_correlation(sig, [flat]) == 0.0


@pytest.mark.phase2
def test_threshold_semantics() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.5])  # almost identical
    # Correlation is very high; default threshold 0.5 rejects.
    assert not is_orthogonal(a, [b])
    # If we widen the threshold above 1, anything is orthogonal.
    assert is_orthogonal(a, [b], threshold=1.5)


@pytest.mark.phase2
def test_max_picks_largest_of_many() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(500)
    b = rng.standard_normal(500)  # near-zero corr
    c = 0.9 * a + 0.1 * rng.standard_normal(500)  # high corr
    assert max_correlation(a, [b, c]) > 0.8
