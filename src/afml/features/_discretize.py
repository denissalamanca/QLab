"""Rolling-quantile discretization (AFML audit V4).

Shannon entropy and Lempel-Ziv complexity require discrete alphabets. The old
``sign(return)`` binary discretization conflates volatility states — a 10-pip
move and a 100-pip move look identical. We instead bin each return into a
**rolling quantile** within its causal look-back window, producing an alphabet
of size ``n_bins`` (default 5 → quintiles) that captures volatility regimes,
not just direction.

This is the single source of truth for the discretization both
``afml.features.shannon`` and ``afml.features.lempel_ziv`` consume.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt

DEFAULT_N_BINS: int = 5  # quintiles


@numba.njit(cache=True)
def rolling_quantile_bin(
    values: npt.NDArray[np.float64],
    window: int,
    n_bins: int,
) -> npt.NDArray[np.int8]:
    """Encode each ``values[t]`` as its quantile bin within a causal window.

    For ``t ≥ window - 1`` the output is the integer in ``[0, n_bins-1]`` such
    that ``values[t]`` falls into that quantile of ``values[t-window+1 : t+1]``.
    Entries with insufficient history are emitted as ``-1`` (sentinel ``NaN``).

    Ties at ``values[t]`` resolve by counting strict inequalities only —
    monotonic sequences always assign the latest element to the top bin.
    """
    n = values.shape[0]
    out = np.full(n, np.int8(-1), dtype=np.int8)
    if window < 2 or n_bins < 2:
        return out

    for t in range(window - 1, n):
        target = values[t]
        if np.isnan(target):
            continue
        rank = 0
        valid = 0
        for i in range(t - window + 1, t + 1):
            ri = values[i]
            if np.isnan(ri):
                continue
            valid += 1
            if ri < target:
                rank += 1
        if valid < 2:
            continue
        # Map rank ∈ [0, valid-1] uniformly onto bin ∈ [0, n_bins-1].
        bin_idx = (rank * n_bins) // valid
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        out[t] = np.int8(bin_idx)
    return out
