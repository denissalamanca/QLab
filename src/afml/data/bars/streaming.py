"""Memory-bounded streaming construction of information bars (Ops M1 hardening).

The single-pass builders in :mod:`tick_imbalance` / :mod:`tick_run` materialise
the *entire* mid-price / volume / timestamp arrays before the numba loop. For a
full-history asset that is fatal: BTCUSD 2020-2025 is ~360 M ticks ⇒ three
~3 GB float64 arrays ⇒ OOM on a 16 GB box.

This module streams the parquet in **row-ordered slices** and runs a *resumable*
kernel that carries the full bar-formation state (EWMAs, running imbalance/run,
the forming bar's OHLCV accumulators, the previous tick price for the sign rule)
across slice boundaries. Output is **bar-for-bar identical** to the single-pass
builders — proven by ``tests/unit/phase1/test_streaming_bars.py`` (exact equality
across every split point) — so the JB bar-type tournament is unchanged; only its
memory profile is bounded.

Row-slice (not time-window) chunking preserves the exact physical tick order the
single-pass kernel sees, so the equality is exact even on an unsorted parquet.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import polars as pl

from afml.data.bars.tick_imbalance import _empty_bar_frame, build_tick_imbalance_bars
from afml.data.bars.tick_run import build_tick_run_bars

# Resumable-kernel state layout (float64 vector; ms timestamps and small ints all
# fit exactly in float64 < 2**53). Shared scalar slots:
_S_EMA_T = 0
_S_EMA_P = 1
_S_LAST_B = 2
_S_BAR_NTICKS = 5
_S_BAR_OPEN_PX = 6
_S_BAR_HIGH = 7
_S_BAR_LOW = 8
_S_BAR_VOL = 9
_S_PREV_PX = 10
_S_HAVE_PREV = 11
_S_BAR_ACTIVE = 12
# TIB-only: slot 3 = theta, slot 4 = bar_plus.
_S_TIB_THETA = 3
_S_TIB_BAR_PLUS = 4
# TRB-only: slot 3 = n_plus, slot 4 = n_minus.
_S_TRB_N_PLUS = 3
_S_TRB_N_MINUS = 4
_STATE_SIZE = 13

DEFAULT_CHUNK_ROWS: int = 5_000_000


def _init_state(initial_expected_ticks: float, initial_prob_buy: float) -> npt.NDArray[np.float64]:
    state = np.zeros(_STATE_SIZE, dtype=np.float64)
    state[_S_EMA_T] = initial_expected_ticks
    state[_S_EMA_P] = initial_prob_buy
    state[_S_LAST_B] = 1.0  # immaterial until the first sign is computed
    return state


@numba.njit(cache=True)
def _tib_chunk(  # noqa: PLR0915 — one sequential state-machine loop; splitting harms clarity
    prices: npt.NDArray[np.float64],
    vols: npt.NDArray[np.float64],
    ts: npt.NDArray[np.int64],
    state: npt.NDArray[np.float64],
    alpha_T: float,
    alpha_P: float,
    min_threshold: float,
    fixed_threshold: float,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    """Resumable Tick-Imbalance-Bar kernel over one row slice. Mutates ``state``."""
    n = prices.shape[0]
    out_ts = np.empty(n, dtype=np.int64)
    out_open = np.empty(n, dtype=np.float64)
    out_high = np.empty(n, dtype=np.float64)
    out_low = np.empty(n, dtype=np.float64)
    out_close = np.empty(n, dtype=np.float64)
    out_vol = np.empty(n, dtype=np.float64)
    out_nticks = np.empty(n, dtype=np.int64)
    k = 0

    ema_T = state[_S_EMA_T]
    ema_P = state[_S_EMA_P]
    last_b = int(state[_S_LAST_B])
    theta = state[_S_TIB_THETA]
    bar_nticks = int(state[_S_BAR_NTICKS])
    bar_plus = int(state[_S_TIB_BAR_PLUS])
    bar_open_px = state[_S_BAR_OPEN_PX]
    bar_high = state[_S_BAR_HIGH]
    bar_low = state[_S_BAR_LOW]
    bar_vol = state[_S_BAR_VOL]
    prev_px = state[_S_PREV_PX]
    have_prev = int(state[_S_HAVE_PREV])
    bar_active = int(state[_S_BAR_ACTIVE])

    for i in range(n):
        px = prices[i]
        v = vols[i]

        if have_prev == 0:
            have_prev = 1
            sign_valid = False
            b = last_b
        else:
            d = px - prev_px
            if d > 0.0:
                b = 1
            elif d < 0.0:
                b = -1
            else:
                b = last_b
            sign_valid = True

        if bar_active == 0:
            bar_open_px = px
            bar_high = px
            bar_low = px
            bar_vol = 0.0
            bar_nticks = 0
            bar_plus = 0
            theta = 0.0
            bar_active = 1

        bar_high = max(bar_high, px)
        bar_low = min(bar_low, px)
        bar_vol += v
        bar_nticks += 1

        if sign_valid:
            last_b = b
            theta += b
            if b == 1:
                bar_plus += 1

        if fixed_threshold > 0.0:
            threshold = fixed_threshold
        else:
            threshold = max(ema_T * abs(2.0 * ema_P - 1.0), min_threshold)

        prev_px = px

        if sign_valid and abs(theta) >= threshold:
            p_in_bar = bar_plus / bar_nticks
            ema_T = alpha_T * bar_nticks + (1.0 - alpha_T) * ema_T
            ema_P = alpha_P * p_in_bar + (1.0 - alpha_P) * ema_P
            out_ts[k] = ts[i]
            out_open[k] = bar_open_px
            out_high[k] = bar_high
            out_low[k] = bar_low
            out_close[k] = px
            out_vol[k] = bar_vol
            out_nticks[k] = bar_nticks
            k += 1
            bar_active = 0

    state[_S_EMA_T] = ema_T
    state[_S_EMA_P] = ema_P
    state[_S_LAST_B] = last_b
    state[_S_TIB_THETA] = theta
    state[_S_BAR_NTICKS] = bar_nticks
    state[_S_TIB_BAR_PLUS] = bar_plus
    state[_S_BAR_OPEN_PX] = bar_open_px
    state[_S_BAR_HIGH] = bar_high
    state[_S_BAR_LOW] = bar_low
    state[_S_BAR_VOL] = bar_vol
    state[_S_PREV_PX] = prev_px
    state[_S_HAVE_PREV] = have_prev
    state[_S_BAR_ACTIVE] = bar_active
    return (
        out_ts[:k],
        out_open[:k],
        out_high[:k],
        out_low[:k],
        out_close[:k],
        out_vol[:k],
        out_nticks[:k],
    )


@numba.njit(cache=True)
def _trb_chunk(  # noqa: PLR0915, PLR0912 — one sequential state-machine loop; splitting harms clarity
    prices: npt.NDArray[np.float64],
    vols: npt.NDArray[np.float64],
    ts: npt.NDArray[np.int64],
    state: npt.NDArray[np.float64],
    alpha_T: float,
    alpha_P: float,
    min_threshold: float,
    fixed_threshold: float,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    """Resumable Tick-Run-Bar kernel over one row slice. Mutates ``state``."""
    n = prices.shape[0]
    out_ts = np.empty(n, dtype=np.int64)
    out_open = np.empty(n, dtype=np.float64)
    out_high = np.empty(n, dtype=np.float64)
    out_low = np.empty(n, dtype=np.float64)
    out_close = np.empty(n, dtype=np.float64)
    out_vol = np.empty(n, dtype=np.float64)
    out_nticks = np.empty(n, dtype=np.int64)
    k = 0

    ema_T = state[_S_EMA_T]
    ema_P = state[_S_EMA_P]
    last_b = int(state[_S_LAST_B])
    n_plus = int(state[_S_TRB_N_PLUS])
    n_minus = int(state[_S_TRB_N_MINUS])
    bar_nticks = int(state[_S_BAR_NTICKS])
    bar_open_px = state[_S_BAR_OPEN_PX]
    bar_high = state[_S_BAR_HIGH]
    bar_low = state[_S_BAR_LOW]
    bar_vol = state[_S_BAR_VOL]
    prev_px = state[_S_PREV_PX]
    have_prev = int(state[_S_HAVE_PREV])
    bar_active = int(state[_S_BAR_ACTIVE])

    for i in range(n):
        px = prices[i]
        v = vols[i]

        if have_prev == 0:
            have_prev = 1
            sign_valid = False
            b = last_b
        else:
            d = px - prev_px
            if d > 0.0:
                b = 1
            elif d < 0.0:
                b = -1
            else:
                b = last_b
            sign_valid = True

        if bar_active == 0:
            bar_open_px = px
            bar_high = px
            bar_low = px
            bar_vol = 0.0
            bar_nticks = 0
            n_plus = 0
            n_minus = 0
            bar_active = 1

        bar_high = max(bar_high, px)
        bar_low = min(bar_low, px)
        bar_vol += v
        bar_nticks += 1

        if sign_valid:
            last_b = b
            if b == 1:
                n_plus += 1
            else:
                n_minus += 1

        theta = n_plus if n_plus >= n_minus else n_minus
        if fixed_threshold > 0.0:
            threshold = fixed_threshold
        else:
            max_prob = ema_P if ema_P >= (1.0 - ema_P) else (1.0 - ema_P)
            threshold = max(ema_T * max_prob, min_threshold)

        prev_px = px

        if sign_valid and theta >= threshold:
            p_in_bar = n_plus / bar_nticks
            ema_T = alpha_T * bar_nticks + (1.0 - alpha_T) * ema_T
            ema_P = alpha_P * p_in_bar + (1.0 - alpha_P) * ema_P
            out_ts[k] = ts[i]
            out_open[k] = bar_open_px
            out_high[k] = bar_high
            out_low[k] = bar_low
            out_close[k] = px
            out_vol[k] = bar_vol
            out_nticks[k] = bar_nticks
            k += 1
            bar_active = 0

    state[_S_EMA_T] = ema_T
    state[_S_EMA_P] = ema_P
    state[_S_LAST_B] = last_b
    state[_S_TRB_N_PLUS] = n_plus
    state[_S_TRB_N_MINUS] = n_minus
    state[_S_BAR_NTICKS] = bar_nticks
    state[_S_BAR_OPEN_PX] = bar_open_px
    state[_S_BAR_HIGH] = bar_high
    state[_S_BAR_LOW] = bar_low
    state[_S_BAR_VOL] = bar_vol
    state[_S_PREV_PX] = prev_px
    state[_S_HAVE_PREV] = have_prev
    state[_S_BAR_ACTIVE] = bar_active
    return (
        out_ts[:k],
        out_open[:k],
        out_high[:k],
        out_low[:k],
        out_close[:k],
        out_vol[:k],
        out_nticks[:k],
    )


_TICK_COLUMNS = ("timestamp", "bid", "ask", "bid_volume", "ask_volume")


def _chunk_arrays(
    chunk: pl.DataFrame,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    mid = ((chunk["bid"] + chunk["ask"]) / 2.0).to_numpy().astype(np.float64)
    vol = (chunk["bid_volume"] + chunk["ask_volume"]).to_numpy().astype(np.float64)
    ts = chunk["timestamp"].cast(pl.Int64).to_numpy().astype(np.int64)
    return mid, vol, ts


def build_information_bars_streaming(
    ticks: pl.LazyFrame,
    *,
    bar_type: str,
    initial_expected_ticks: float = 100.0,
    initial_prob_buy: float = 0.5,
    alpha_T: float = 0.05,
    alpha_P: float = 0.05,
    min_threshold: float = 1.0,
    fixed_threshold: float = 0.0,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> pl.DataFrame:
    """Build TIB or TRB bars from a lazy tick frame in bounded memory.

    Streams ``ticks`` in ``chunk_rows``-row physical slices, carrying full
    bar-formation state across boundaries, so the output is identical to the
    single-pass builder but never materialises more than one slice of ticks.

    Parameters
    ----------
    ticks
        A lazy tick frame (e.g. from :func:`afml.data.load_ticks`) with the
        standard ``timestamp/bid/ask/bid_volume/ask_volume`` schema.
    bar_type
        ``"tick_imbalance"`` or ``"tick_run"``.
    initial_expected_ticks, initial_prob_buy, alpha_T, alpha_P, min_threshold
        Identical semantics to the single-pass builders.
    fixed_threshold
        ``> 0`` pins the imbalance/run threshold to a constant (the
        target-calibrated mode — robust against the adaptive EWMA's
        runaway-collapse; see :func:`calibrate_information_bar_threshold`).
    chunk_rows
        Physical slice size. Memory ≈ ``chunk_rows × ~40 bytes``.
    """
    if bar_type not in ("tick_imbalance", "tick_run"):
        raise ValueError(f"bar_type must be tick_imbalance|tick_run, got {bar_type!r}")

    projected = ticks.select(list(_TICK_COLUMNS))
    state = _init_state(initial_expected_ticks, initial_prob_buy)
    cols: list[list[npt.NDArray[np.generic]]] = [[] for _ in range(7)]
    offset = 0
    while True:
        chunk = projected.slice(offset, chunk_rows).collect()
        height = chunk.height
        if height == 0:
            break
        mid, vol, ts = _chunk_arrays(chunk)
        kernel = _tib_chunk if bar_type == "tick_imbalance" else _trb_chunk
        emitted = kernel(mid, vol, ts, state, alpha_T, alpha_P, min_threshold, fixed_threshold)
        if emitted[0].size:
            for j in range(7):
                cols[j].append(emitted[j])
        offset += height
        if height < chunk_rows:
            break

    if not cols[0]:
        return _empty_bar_frame()

    ts_all = np.concatenate(cols[0])
    return pl.DataFrame({
        "timestamp": pl.Series(ts_all).cast(pl.Datetime("ms", "UTC")),
        "open": np.concatenate(cols[1]),
        "high": np.concatenate(cols[2]),
        "low": np.concatenate(cols[3]),
        "close": np.concatenate(cols[4]),
        "volume": np.concatenate(cols[5]),
        "n_ticks": np.concatenate(cols[6]),
    })


def calibrate_information_bar_threshold(
    sample: pl.DataFrame,
    *,
    bar_type: str,
    target_ticks_per_bar: float,
    max_iter: int = 40,
    tol: float = 0.03,
) -> float:
    """Bisect a *fixed* threshold ``h`` so the sample averages ≈ ``target_ticks_per_bar``.

    The adaptive EWMA threshold runs away (collapses to ~1 tick/bar over long
    spans). A fixed ``h`` calibrated so a representative tick **sample** hits the
    regime's target ticks/bar yields ≈ that granularity on the full series, with
    no self-adaptation to collapse. ``n_bars(h)`` is monotone-decreasing, so a
    simple bisection converges. Run on a *contiguous* sample (the tick-sign rule
    depends on consecutive ticks) — striding would corrupt it.

    Returns the calibrated ``h`` (pass as ``fixed_threshold=`` to the builders).
    """
    if bar_type not in ("tick_imbalance", "tick_run"):
        raise ValueError(f"bar_type must be tick_imbalance|tick_run, got {bar_type!r}")
    builder = build_tick_imbalance_bars if bar_type == "tick_imbalance" else build_tick_run_bars
    n = sample.height
    if n == 0 or target_ticks_per_bar <= 0.0:
        return 1.0
    target_bars = max(float(n) / target_ticks_per_bar, 1.0)

    lo, hi = 1.0, max(target_ticks_per_bar, 2.0)
    best = hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        bars = builder(sample, fixed_threshold=mid).height
        best = mid
        if abs(bars - target_bars) <= tol * target_bars:
            break
        if bars > target_bars:  # too many bars → raise the threshold
            lo = mid
        else:  # too few bars → lower the threshold
            hi = mid
    return best
