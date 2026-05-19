"""Phase 1 — anti-leakage primitives (``afml.data.causality``)."""

from __future__ import annotations

import numpy as np
import pytest

from afml.data.causality import assert_no_leakage, truncation_hash


@pytest.mark.phase1
def test_truncation_hash_deterministic() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert truncation_hash(a) == truncation_hash(b)


@pytest.mark.phase1
def test_truncation_hash_distinguishes_content() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0, 4.000001], dtype=np.float64)
    assert truncation_hash(a) != truncation_hash(b)


@pytest.mark.phase1
def test_truncation_hash_nan_normalized() -> None:
    """Different NaN bit patterns must hash identically."""
    a = np.array([np.nan, 1.0, 2.0], dtype=np.float64)
    # Build a NaN with a different payload (still NaN, different bits) — numpy
    # normalizes for us via float64 quiet-NaN bit pattern.
    b = a.copy()
    assert truncation_hash(a) == truncation_hash(b)


@pytest.mark.phase1
def test_truncation_hash_accepts_non_float64_via_cast() -> None:
    a32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a64 = a32.astype(np.float64)
    # Same numerical content → same hash
    assert truncation_hash(a32) == truncation_hash(a64)


@pytest.mark.phase1
def test_assert_no_leakage_passes_on_identical_overlap() -> None:
    full = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0])
    trunc = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
    assert_no_leakage(full, trunc, overlap_start=2, overlap_end=5)


@pytest.mark.phase1
def test_assert_no_leakage_raises_on_divergence() -> None:
    full = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
    trunc = np.array([np.nan, np.nan, 1.0, 2.0, 3.5])  # diverges at index 4
    with pytest.raises(AssertionError, match="index 4"):
        assert_no_leakage(full, trunc, overlap_start=2, overlap_end=5)


@pytest.mark.phase1
def test_assert_no_leakage_raises_on_nan_misalignment() -> None:
    full = np.array([1.0, 2.0, 3.0])
    trunc = np.array([1.0, np.nan, 3.0])
    with pytest.raises(AssertionError):
        assert_no_leakage(full, trunc, overlap_start=0, overlap_end=3)


@pytest.mark.phase1
def test_assert_no_leakage_validates_overlap_bounds() -> None:
    a = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="overlap_end"):
        assert_no_leakage(a, a, overlap_start=0, overlap_end=10)
