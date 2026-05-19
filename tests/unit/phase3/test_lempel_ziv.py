"""Phase 3 — Lempel-Ziv Complexity."""

from __future__ import annotations

import numpy as np
import pytest

from afml.features.lempel_ziv import _lz_complexity_binary, lempel_ziv_complexity


@pytest.mark.phase3
def test_lz_complexity_relative_ordering() -> None:
    """LZ counters: zeros < alternating < random. The absolute value depends
    on the LZ76 trailing-phrase convention; what matters for downstream ML
    is the *ordering*, which is stable across conventions."""
    zeros = np.zeros(64, dtype=np.int8)
    alt = np.tile([0, 1], 32).astype(np.int8)
    rng = np.random.default_rng(0)
    rnd = (rng.standard_normal(64) > 0).astype(np.int8)

    c_zeros = _lz_complexity_binary(zeros)
    c_alt = _lz_complexity_binary(alt)
    c_rnd = _lz_complexity_binary(rnd)

    # Strictly increasing in disorder.
    assert c_zeros < c_alt < c_rnd
    # All small positive integers (sanity).
    assert 1 <= c_zeros <= 4
    assert c_rnd <= 64  # can never exceed sequence length


@pytest.mark.phase3
def test_lz_length_preserved() -> None:
    close = np.cumsum(np.random.default_rng(0).standard_normal(500)) + 100.0
    out = lempel_ziv_complexity(close, window=50)
    assert out.shape == close.shape


@pytest.mark.phase3
def test_lz_low_on_monotonic() -> None:
    close = 100.0 + np.arange(500) * 0.1
    out = lempel_ziv_complexity(close, window=50)
    finite = out[np.isfinite(out)]
    # Monotonic up ⇒ all signs equal '1' ⇒ minimum LZ.
    # Normalized LZ for an all-ones sequence is small (a couple distinct
    # patterns spread over log2(50)/50 ≈ 0.113).
    assert np.all(finite < 0.5)


@pytest.mark.phase3
def test_lz_higher_on_random_than_monotonic() -> None:
    rng = np.random.default_rng(0)
    rnd_close = 100.0 + np.cumsum(rng.standard_normal(1000) * 0.05)
    mono_close = 100.0 + np.arange(1000) * 0.1

    rnd_lz = lempel_ziv_complexity(rnd_close, window=100)
    mono_lz = lempel_ziv_complexity(mono_close, window=100)

    rnd_finite = rnd_lz[np.isfinite(rnd_lz)]
    mono_finite = mono_lz[np.isfinite(mono_lz)]
    assert float(np.mean(rnd_finite)) > float(np.mean(mono_finite))


@pytest.mark.phase3
def test_lz_rejects_tiny_window() -> None:
    with pytest.raises(ValueError, match="window"):
        lempel_ziv_complexity(np.zeros(100), window=3)
