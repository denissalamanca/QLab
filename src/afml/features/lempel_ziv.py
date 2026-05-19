"""Lempel-Ziv (1976) complexity of return-quantile sequences (AFML audit V4).

We bin each return into a rolling-quantile alphabet of size ``n_bins`` (default
5 → quintiles), then count distinct LZ phrases as the window slides. The
binary-direction encoding that we previously used conflated volatility states
— quantile binning preserves both direction and magnitude information.

Output is window-normalized by ``log_b(window) / window`` (Kaspar & Schuster
1987) where ``b = n_bins``. Strictly causal via ``.shift(1)``.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd

from afml.features._discretize import DEFAULT_N_BINS, rolling_quantile_bin


@numba.njit(cache=True)
def _lz_complexity_binary(seq: npt.NDArray[np.int8]) -> int:
    """Lempel-Ziv 1976 complexity counter for a small-alphabet integer sequence.

    Works on any ``int8`` alphabet (binary {0,1}, quintiles {0..4}, etc.). The
    parser is alphabet-agnostic — it only counts distinct substring patterns.
    """
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
def _rolling_lz_from_bins(
    bins: npt.NDArray[np.int8],
    window: int,
    n_bins: int,
) -> npt.NDArray[np.float64]:
    n = bins.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    # Normalize by log_b(window) / window for window/alphabet invariance.
    norm = np.log(window) / (np.log(n_bins) * window) if (window > 1 and n_bins > 1) else 1.0
    for t in range(window - 1, n):
        # Skip windows containing the sentinel -1 (insufficient history).
        valid = True
        for i in range(t - window + 1, t + 1):
            if bins[i] < 0:
                valid = False
                break
        if not valid:
            continue
        sub = bins[t - window + 1 : t + 1]
        c = _lz_complexity_binary(sub)
        out[t] = c * norm
    return out


def lempel_ziv_complexity(
    close: npt.NDArray[np.float64],
    *,
    window: int = 50,
    n_bins: int = DEFAULT_N_BINS,
) -> npt.NDArray[np.float64]:
    """Rolling normalized LZ complexity over quantile-binned returns."""
    if close.ndim != 1:
        raise ValueError(f"close must be 1-D; got shape {close.shape}")
    if window < 4:
        raise ValueError(f"window must be >= 4 for meaningful LZ; got {window}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")

    close_f = close.astype(np.float64, copy=False)
    log_close = np.log(close_f)
    returns = np.empty_like(log_close)
    returns[0] = np.nan
    returns[1:] = np.diff(log_close)

    bins = rolling_quantile_bin(returns, window, n_bins)
    raw = _rolling_lz_from_bins(bins, window, n_bins)
    return pd.Series(raw).shift(1).to_numpy(dtype=np.float64)
