"""Shannon Entropy — rolling entropy of return-quantile sequences (audit V4).

Returns are bucketed into a rolling-quantile alphabet of size ``n_bins``
(default 5 → quintiles). Entropy is computed in bits:
    H = -Σ p_i · log₂(p_i),   i ∈ {0, 1, ..., n_bins - 1}

Maximum entropy with ``n_bins = 5`` is ``log₂(5) ≈ 2.32`` bits. Higher H ⇒
more disordered local price action (closer to noise). Quantile binning captures
volatility state, not just direction — a 10-pip move and a 100-pip move land
in different bins when the surrounding window includes both magnitudes (AFML
audit V4).

Strictly causal via ``.shift(1)``.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd

from afml.features._discretize import DEFAULT_N_BINS, rolling_quantile_bin


@numba.njit(cache=True)
def _rolling_entropy_from_bins(
    bins: npt.NDArray[np.int8],
    window: int,
    n_bins: int,
) -> npt.NDArray[np.float64]:
    n = bins.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for t in range(window - 1, n):
        # Reset and recount within the window.
        for k in range(n_bins):
            counts[k] = 0
        valid = 0
        for i in range(t - window + 1, t + 1):
            b = bins[i]
            if b < 0:
                continue
            counts[b] += 1
            valid += 1
        if valid == 0:
            continue
        h = 0.0
        for k in range(n_bins):
            c = counts[k]
            if c > 0:
                p = c / valid
                h -= p * np.log2(p)
        out[t] = h
    return out


def shannon_entropy(
    close: npt.NDArray[np.float64],
    *,
    window: int = 50,
    n_bins: int = DEFAULT_N_BINS,
) -> npt.NDArray[np.float64]:
    """Rolling Shannon entropy (bits) of quantile-binned returns.

    Parameters
    ----------
    close : 1-D bar close prices.
    window : look-back in bars for the quantile binning AND for the entropy
        estimate. The same window is used for both; passing a different value
        here would mean re-discretizing on a different look-back than the
        entropy considers, which is rarely meaningful.
    n_bins : alphabet size for quantile binning. ``5`` (quintiles) captures
        volatility states without over-fragmenting the histogram.
    """
    if close.ndim != 1:
        raise ValueError(f"close must be 1-D; got shape {close.shape}")
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")

    close_f = close.astype(np.float64, copy=False)
    log_close = np.log(close_f)
    returns = np.empty_like(log_close)
    returns[0] = np.nan
    returns[1:] = np.diff(log_close)

    bins = rolling_quantile_bin(returns, window, n_bins)
    raw = _rolling_entropy_from_bins(bins, window, n_bins)
    return pd.Series(raw).shift(1).to_numpy(dtype=np.float64)
