"""Anti-leakage primitives shared across Phases 1 and 3.

The ``truncation hash test`` is the mathematical guarantee that a rolling
computation (FFD in Phase 1, microstructure features in Phase 3) uses only
strictly-past data. Concretely: computing the rolling transform on a full series
[0, N) and on a truncated series [0, M) (M < N) must produce *byte-identical*
output for all valid overlapping rows.

If they differ, the implementation is leaking future information into the past —
fatal for a real trading system.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import numpy.typing as npt


def truncation_hash(values: npt.NDArray[Any]) -> str:
    """Deterministic SHA-256 of a numpy array's content.

    NaN entries are normalized to a single canonical NaN bit pattern so that
    "any NaN equals any NaN" for hashing purposes. The dtype (``float64``) is
    fixed to avoid endianness or precision drift.
    """
    if values.dtype != np.float64:
        values = values.astype(np.float64, copy=False)
    canonical = np.where(np.isnan(values), np.float64("nan"), values)
    # numpy's NaN bit pattern is platform-stable for float64; .tobytes() is
    # endianness-aware but deterministic on a single architecture.
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def assert_no_leakage(
    full: npt.NDArray[np.float64],
    truncated: npt.NDArray[np.float64],
    *,
    overlap_start: int,
    overlap_end: int,
) -> None:
    """Verify two rolling-transform outputs agree exactly on their overlap.

    Both arrays must have the same prefix up to ``overlap_end``. NaNs are matched
    by position. Raises ``AssertionError`` with the divergent index if any
    overlapping entry differs.

    Parameters
    ----------
    full : array computed on the full input series
    truncated : array computed on the truncated (shorter) input series
    overlap_start : first valid output index (typically the warm-up window
        length minus one; entries before this should be NaN in both arrays)
    overlap_end : exclusive end of the overlap (= length of the truncated input)
    """
    if overlap_end > len(full) or overlap_end > len(truncated):
        raise ValueError(
            f"overlap_end={overlap_end} exceeds array lengths "
            f"(full={len(full)}, truncated={len(truncated)})"
        )
    a = full[overlap_start:overlap_end]
    b = truncated[overlap_start:overlap_end]
    h_a = truncation_hash(a)
    h_b = truncation_hash(b)
    if h_a != h_b:
        # Pinpoint the first divergence for an actionable error message.
        for offset, (x, y) in enumerate(zip(a, b, strict=True)):
            x_nan, y_nan = np.isnan(x), np.isnan(y)
            if x_nan != y_nan or (not x_nan and x != y):
                raise AssertionError(
                    f"Future-data leakage detected at output index "
                    f"{overlap_start + offset}: full={x!r} vs truncated={y!r}"
                )
        raise AssertionError(
            f"Hashes differ ({h_a[:12]}… vs {h_b[:12]}…) "
            "but element-wise scan found no divergence — possible dtype mismatch"
        )
