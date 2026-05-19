"""Lempel-Ziv (1976) complexity of return-sign binary sequences.

LZ counts the number of distinct substrings appearing as the sequence is
parsed left-to-right — a measure of algorithmic randomness. We binarize the
price-change sign sequence (``+1 → '1'``, ``≤ 0 → '0'``), compute LZ on a
rolling window, and normalize by ``log₂(n) / n`` so the value is window-
length invariant (Kaspar & Schuster 1987).

Strictly causal via ``.shift(1)``.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd


@numba.njit(cache=True)
def _lz_complexity_binary(seq: npt.NDArray[np.int8]) -> int:
    """Lempel-Ziv 1976 complexity counter for a binary sequence."""
    n = seq.shape[0]
    if n == 0:
        return 0
    i = 0
    c = 1
    length = 1
    while i + length <= n:
        # Is seq[i : i+length] a substring of seq[0 : i+length-1]?
        found = False
        prefix_end = i + length - 1
        max_start = prefix_end - length
        if max_start >= 0:
            for start in range(max_start + 1):
                match = True
                for k in range(length):
                    if seq[start + k] != seq[i + k]:
                        match = False
                        break
                if match:
                    found = True
                    break
        if found:
            length += 1
            if i + length > n:
                c += 1
                break
        else:
            c += 1
            i += length
            length = 1
            if i >= n:
                break
    return c


@numba.njit(cache=True)
def _rolling_lz(signs: npt.NDArray[np.int8], window: int) -> npt.NDArray[np.float64]:
    n = signs.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    norm = np.log2(window) / window if window > 1 else 1.0
    for t in range(window - 1, n):
        sub = signs[t - window + 1 : t + 1]
        c = _lz_complexity_binary(sub)
        out[t] = c * norm
    return out


def lempel_ziv_complexity(
    close: npt.NDArray[np.float64], *, window: int = 50
) -> npt.NDArray[np.float64]:
    """Rolling normalized LZ complexity of binary return signs."""
    if close.ndim != 1:
        raise ValueError(f"close must be 1-D; got shape {close.shape}")
    if window < 4:
        raise ValueError(f"window must be >= 4 for meaningful LZ; got {window}")

    dp = np.diff(close, prepend=close[0])
    signs = (dp > 0.0).astype(np.int8)
    raw = _rolling_lz(signs, window)
    return pd.Series(raw).shift(1).to_numpy(dtype=np.float64)
