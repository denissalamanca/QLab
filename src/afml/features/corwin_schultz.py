"""Corwin-Schultz (2012) Bid-Ask Spread estimator — time-adjusted variant (audit V2).

Classical Corwin-Schultz assumes equal-duration bars: variance scales linearly
in time with a fixed Δt per bar. On Phase 1 Information Bars (TIB / TRB) the
Δt between consecutive bars is **stochastic**, so the formula's Brownian-motion
assumption ("2 bars have twice the variance of 1 bar") is mathematically
destroyed if applied naively.

Fix per AFML audit Vulnerability 2: normalize the squared log range
``ln²(H/L)`` by each bar's actual Δt (chronological time elapsed) BEFORE
combining bars. The β and γ inputs to the formula then represent per-unit-time
variance estimates that satisfy the original derivation regardless of bar
duration. The classical (3 - 2√2) denominator constants remain valid in this
time-normalized regime.

When ``bar_durations`` is not supplied the function degrades gracefully to the
classical formula (treating every bar as unit-time) — preserving the
time-bar use-case.

Strictly causal via ``.shift(1)`` on the spread output.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def corwin_schultz_spread(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    *,
    bar_durations: npt.NDArray[np.float64] | None = None,
    window: int = 2,
) -> npt.NDArray[np.float64]:
    """Rolling Corwin-Schultz spread, time-adjusted for stochastic Δt bars.

    Parameters
    ----------
    high, low : 1-D arrays of bar highs / lows (same length).
    bar_durations : 1-D array of bar durations in seconds (or any consistent
        unit). When omitted, every bar is treated as unit-time (the classical
        regime, valid for time bars).
    window : kept for API symmetry; rolling-mean smoothing across larger
        ``window`` values uses the 2-bar estimates as building blocks.

    Returns
    -------
    1-D ``float64`` array of the same length. NaN warm-up at the head.
    """
    if high.shape != low.shape:
        raise ValueError("high and low must have the same shape")
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")
    if bar_durations is not None and bar_durations.shape != high.shape:
        raise ValueError("bar_durations must have the same shape as high/low")

    h = pd.Series(high.astype(np.float64))
    l_ = pd.Series(low.astype(np.float64))

    # Per-bar squared log range.
    log_hl_sq = np.log(h / l_) ** 2

    # Time-normalize: ln²(H/L) per unit Δt. Under pure Brownian motion this is
    # constant ∝ σ², regardless of how long the bar lasted. Falls back to unit
    # Δt when durations aren't supplied.
    if bar_durations is not None:
        dt = pd.Series(bar_durations.astype(np.float64))
        # Guard against zero / negative Δt (degenerate bars).
        dt_safe = dt.where(dt > 0.0, np.nan)
        log_hl_sq_per_dt = log_hl_sq / dt_safe
    else:
        log_hl_sq_per_dt = log_hl_sq
        dt = pd.Series(np.ones_like(high, dtype=np.float64))

    # β = per-unit-time squared log range, summed over 2 consecutive bars.
    beta = log_hl_sq_per_dt + log_hl_sq_per_dt.shift(1)

    # γ = squared log range over the COMBINED 2-bar window, also per-unit-time.
    h_prev = h.shift(1)
    l_prev = l_.shift(1)
    h2 = np.maximum(h, h_prev)
    l2 = np.minimum(l_, l_prev)
    combined_dt = dt + dt.shift(1)
    combined_dt_safe = combined_dt.where(combined_dt > 0.0, np.nan)
    gamma = (np.log(h2 / l2) ** 2) / combined_dt_safe

    # Standard C-S algebra, now operating on time-normalized inputs.
    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    raw_spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    raw_spread = raw_spread.clip(lower=0.0)

    if window > 2:
        smoothed = raw_spread.rolling(window=window, min_periods=window).mean()
    else:
        smoothed = raw_spread
    return smoothed.shift(1).to_numpy(dtype=np.float64)
