"""Corwin-Schultz (2012) Bid-Ask Spread estimator.

Uses two consecutive bars' high / low to infer the effective spread without
quote data:

    β = ln(H_{t-1}/L_{t-1})² + ln(H_t/L_t)²
    H₂ = max(H_{t-1}, H_t),  L₂ = min(L_{t-1}, L_t)
    γ = ln(H₂/L₂)²
    α = (√(2β) - √β) / (3 - 2√2)  -  √(γ / (3 - 2√2))
    spread = max(0, 2·(e^α - 1) / (1 + e^α))

Implementation shifts the final spread by one bar so the value at ``t`` uses
only bars ``t-1`` and ``t-2`` (strictly causal at the event timestamp).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def corwin_schultz_spread(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    *,
    window: int = 2,
) -> npt.NDArray[np.float64]:
    """Rolling Corwin-Schultz spread estimate.

    Parameters
    ----------
    high, low : 1-D arrays of bar highs / lows (same length).
    window : kept for API symmetry; the estimator uses 2 consecutive bars,
        but a larger ``window`` smooths the output via a rolling mean of the
        2-bar estimates.

    Returns
    -------
    1-D ``float64`` array of the same length. NaN warm-up at the head.
    """
    if high.shape != low.shape:
        raise ValueError("high and low must have the same shape")
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")

    h = pd.Series(high.astype(np.float64))
    l_ = pd.Series(low.astype(np.float64))

    log_hl_sq = np.log(h / l_) ** 2
    h_prev = h.shift(1)
    l_prev = l_.shift(1)
    h2 = np.maximum(h, h_prev)
    l2 = np.minimum(l_, l_prev)
    gamma = np.log(h2 / l2) ** 2
    beta = log_hl_sq + log_hl_sq.shift(1)

    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    raw_spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    raw_spread = raw_spread.clip(lower=0.0)

    # Optional smoothing across larger window, then strict causality shift.
    if window > 2:
        smoothed = raw_spread.rolling(window=window, min_periods=window).mean()
    else:
        smoothed = raw_spread
    return smoothed.shift(1).to_numpy(dtype=np.float64)
