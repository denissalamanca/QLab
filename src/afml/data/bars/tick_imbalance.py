"""Tick Imbalance Bars (TIB) — Blueprint §3.1, López de Prado 2018 §2.3.2.

Algorithm
---------
Tick sign ``b_t`` is +1 if the trade was at a higher price than the previous,
-1 if lower, and carries forward on equal-price ticks.

Cumulative imbalance: ``θ_t = Σ_{i ≤ t} b_i`` (reset at each bar).

A new bar is sampled when ``|θ_t| ≥ E_0[T] · |2 P[b_t = +1] − 1|``, where:
- ``E_0[T]``  = EWMA of ticks-per-bar from **previously completed bars only**.
- ``P[b_t = +1]`` = EWMA of the fraction of +1 ticks from previously closed bars.

**Causal-update invariant (AFML audit Vulnerability 3):**
The threshold a forming bar k sees is fully determined at the close of bar
k-1 — both EMAs are last updated at that point and are read-only during bar k.
EMA values are advanced ONLY at the moment a bar closes, using that bar's
just-finalized statistics; bar k+1 then sees the newly-updated EMAs.
``tests/unit/phase1/test_tick_imbalance.py::test_tib_truncation_invariance``
proves this by checking that bar closes computed on a full tick series match
those computed on a truncated prefix for every bar that closes before the
truncation point.

Initial values are deliberately chosen so the first few bars warm up the EWMAs
quickly without runaway behavior. There are no hard-coded scale parameters —
``alpha_T`` and ``alpha_P`` are configurable but the algorithm self-adapts.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import polars as pl


@numba.njit(cache=True)
def _compute_tib_bar_indices(
    prices: npt.NDArray[np.float64],
    initial_expected_ticks: float,
    initial_prob_buy: float,
    alpha_T: float,
    alpha_P: float,
    min_threshold: float,
) -> npt.NDArray[np.int64]:
    """Return the closing-tick index of each TIB bar."""
    n = prices.shape[0]
    closes = np.empty(n, dtype=np.int64)
    n_bars = 0

    ema_T = initial_expected_ticks
    ema_P = initial_prob_buy

    last_b = 1  # initial direction (immaterial after first bar)
    theta = 0.0
    bar_open = 0
    bar_plus_count = 0

    for i in range(1, n):
        d = prices[i] - prices[i - 1]
        if d > 0.0:
            b = 1
        elif d < 0.0:
            b = -1
        else:
            b = last_b
        last_b = b

        theta += b
        if b == 1:
            bar_plus_count += 1

        threshold = ema_T * abs(2.0 * ema_P - 1.0)
        threshold = max(threshold, min_threshold)

        if abs(theta) >= threshold:
            ticks_in_bar = i - bar_open + 1
            p_in_bar = bar_plus_count / ticks_in_bar

            # Update EWMAs.
            ema_T = alpha_T * ticks_in_bar + (1.0 - alpha_T) * ema_T
            ema_P = alpha_P * p_in_bar + (1.0 - alpha_P) * ema_P

            closes[n_bars] = i
            n_bars += 1

            bar_open = i + 1
            theta = 0.0
            bar_plus_count = 0

    return closes[:n_bars]


@numba.njit(cache=True)
def _aggregate_ohlc(
    prices: npt.NDArray[np.float64],
    volumes: npt.NDArray[np.float64],
    closes: npt.NDArray[np.int64],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    """Compute OHLC + volume + n_ticks given bar close-indices."""
    n_bars = closes.shape[0]
    opens_idx = np.empty(n_bars, dtype=np.int64)
    open_p = np.empty(n_bars, dtype=np.float64)
    high_p = np.empty(n_bars, dtype=np.float64)
    low_p = np.empty(n_bars, dtype=np.float64)
    close_p = np.empty(n_bars, dtype=np.float64)
    vol = np.empty(n_bars, dtype=np.float64)
    n_ticks = np.empty(n_bars, dtype=np.int64)

    start = 0
    for b in range(n_bars):
        end = closes[b]
        opens_idx[b] = start
        hi = prices[start]
        lo = prices[start]
        v = 0.0
        for i in range(start, end + 1):
            p = prices[i]
            hi = max(hi, p)
            lo = min(lo, p)
            v += volumes[i]
        open_p[b] = prices[start]
        high_p[b] = hi
        low_p[b] = lo
        close_p[b] = prices[end]
        vol[b] = v
        n_ticks[b] = end - start + 1
        start = end + 1

    return opens_idx, open_p, high_p, low_p, close_p, vol, n_ticks


def build_tick_imbalance_bars(
    ticks: pl.DataFrame,
    *,
    initial_expected_ticks: float = 100.0,
    initial_prob_buy: float = 0.5,
    alpha_T: float = 0.05,
    alpha_P: float = 0.05,
    min_threshold: float = 1.0,
) -> pl.DataFrame:
    """Build Tick Imbalance Bars from a tick DataFrame.

    Parameters
    ----------
    ticks : Polars DataFrame with ``timestamp``, ``bid``, ``ask``,
        ``bid_volume``, ``ask_volume``. Mid-price (= 0.5·(bid+ask)) drives the
        tick direction; total volume = bid_volume + ask_volume.
    initial_expected_ticks, initial_prob_buy : warm-up EWMA values. Their
        influence vanishes after a few hundred bars. They are NOT free
        hyperparameters in the AFML sense — the EWMAs self-adapt.
    alpha_T, alpha_P : EWMA decay constants for the expected-ticks and
        probability estimates. Smaller ⇒ slower adaptation.
    min_threshold : floor on the imbalance threshold to prevent pathological
        bar-after-every-tick behavior when ``|2P-1|`` collapses to ~0.

    Returns
    -------
    DataFrame with columns ``timestamp`` (bar close), ``open``, ``high``,
    ``low``, ``close``, ``volume``, ``n_ticks``.
    """
    if ticks.height == 0:
        return _empty_bar_frame()

    mid = ((ticks["bid"] + ticks["ask"]) / 2.0).to_numpy().astype(np.float64)
    total_vol = (ticks["bid_volume"] + ticks["ask_volume"]).to_numpy().astype(np.float64)
    ts = ticks["timestamp"].to_numpy()

    closes = _compute_tib_bar_indices(
        mid,
        initial_expected_ticks,
        initial_prob_buy,
        alpha_T,
        alpha_P,
        min_threshold,
    )
    if closes.size == 0:
        return _empty_bar_frame()

    _, open_p, high_p, low_p, close_p, vol, n_ticks = _aggregate_ohlc(mid, total_vol, closes)

    return pl.DataFrame({
        "timestamp": ts[closes],
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "volume": vol,
        "n_ticks": n_ticks,
    })


def _empty_bar_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("ms", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "n_ticks": pl.Int64,
        }
    )
