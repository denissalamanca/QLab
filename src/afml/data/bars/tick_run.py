"""Tick Run Bars (TRB) — Blueprint §3.1, López de Prado 2018 §2.3.2.

Algorithm
---------
Tick sign ``b_t`` is +1 / −1 / carry-forward exactly as in TIB.

Run statistic: ``θ_t = max(N⁺_t, N⁻_t)`` where ``N⁺_t`` / ``N⁻_t`` are the
cumulative counts of +1 / −1 ticks since the previous bar.

A new bar is sampled when ``θ_t ≥ E_0[T] · max(P⁺, P⁻)``, with EWMA estimates
of ``E_0[T]`` (expected ticks per bar) and ``P⁺`` = P[b = +1].

TRB is most informative in momentum regimes; combined with TIB and time bars
in the Jarque-Bera tournament to pick the best sampling for each instrument.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import polars as pl

from afml.data.bars.tick_imbalance import _aggregate_ohlc, _empty_bar_frame


@numba.njit(cache=True)
def _compute_trb_bar_indices(
    prices: npt.NDArray[np.float64],
    initial_expected_ticks: float,
    initial_prob_buy: float,
    alpha_T: float,
    alpha_P: float,
    min_threshold: float,
) -> npt.NDArray[np.int64]:
    """Return the closing-tick index of each TRB bar."""
    n = prices.shape[0]
    closes = np.empty(n, dtype=np.int64)
    n_bars = 0

    ema_T = initial_expected_ticks
    ema_P = initial_prob_buy

    last_b = 1
    n_plus = 0
    n_minus = 0
    bar_open = 0

    for i in range(1, n):
        d = prices[i] - prices[i - 1]
        if d > 0.0:
            b = 1
        elif d < 0.0:
            b = -1
        else:
            b = last_b
        last_b = b

        if b == 1:
            n_plus += 1
        else:
            n_minus += 1

        theta = n_plus if n_plus >= n_minus else n_minus

        # Threshold: expected-ticks scaled by max probability.
        max_prob = ema_P if ema_P >= (1.0 - ema_P) else (1.0 - ema_P)
        threshold = ema_T * max_prob
        threshold = max(threshold, min_threshold)

        if theta >= threshold:
            ticks_in_bar = i - bar_open + 1
            p_in_bar = n_plus / ticks_in_bar

            ema_T = alpha_T * ticks_in_bar + (1.0 - alpha_T) * ema_T
            ema_P = alpha_P * p_in_bar + (1.0 - alpha_P) * ema_P

            closes[n_bars] = i
            n_bars += 1

            bar_open = i + 1
            n_plus = 0
            n_minus = 0

    return closes[:n_bars]


def build_tick_run_bars(
    ticks: pl.DataFrame,
    *,
    initial_expected_ticks: float = 100.0,
    initial_prob_buy: float = 0.5,
    alpha_T: float = 0.05,
    alpha_P: float = 0.05,
    min_threshold: float = 1.0,
) -> pl.DataFrame:
    """Build Tick Run Bars from a tick DataFrame.

    See ``build_tick_imbalance_bars`` for the parameter semantics; both bar
    types share the same EWMA framework but trigger on different statistics
    of the tick-sign sequence.
    """
    if ticks.height == 0:
        return _empty_bar_frame()

    mid = ((ticks["bid"] + ticks["ask"]) / 2.0).to_numpy().astype(np.float64)
    total_vol = (ticks["bid_volume"] + ticks["ask_volume"]).to_numpy().astype(np.float64)
    ts = ticks["timestamp"].to_numpy()

    closes = _compute_trb_bar_indices(
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
