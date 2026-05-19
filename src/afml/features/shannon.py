"""Shannon Entropy — rolling entropy of return-sign trinary sequence.

Returns are binned into {+1, 0, -1}. Entropy is computed in bits:
    H = -Σ p_i · log₂(p_i),  i ∈ {+, 0, -}

Higher H ⇒ more disordered local price action (closer to noise). Strictly
causal via ``.shift(1)``.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd


@numba.njit(cache=True)
def _rolling_entropy_signs(signs: npt.NDArray[np.int8], window: int) -> npt.NDArray[np.float64]:
    n = signs.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(window - 1, n):
        n_pos = 0
        n_neg = 0
        n_zero = 0
        for i in range(t - window + 1, t + 1):
            s = signs[i]
            if s == 1:
                n_pos += 1
            elif s == -1:
                n_neg += 1
            else:
                n_zero += 1
        h = 0.0
        for cnt in (n_pos, n_neg, n_zero):
            if cnt > 0:
                p = cnt / window
                h -= p * np.log2(p)
        out[t] = h
    return out


def shannon_entropy(close: npt.NDArray[np.float64], *, window: int = 50) -> npt.NDArray[np.float64]:
    """Rolling Shannon entropy (bits) of trinary return signs."""
    if close.ndim != 1:
        raise ValueError(f"close must be 1-D; got shape {close.shape}")
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")

    dp = np.diff(close, prepend=close[0])
    signs = np.zeros_like(dp, dtype=np.int8)
    signs[dp > 0.0] = 1
    signs[dp < 0.0] = -1
    raw = _rolling_entropy_signs(signs, window)
    return pd.Series(raw).shift(1).to_numpy(dtype=np.float64)
