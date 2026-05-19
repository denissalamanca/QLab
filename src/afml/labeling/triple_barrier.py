"""Triple-Barrier labeling (Blueprint §4.2 + López de Prado 2018 Ch. 3).

For each Brain 1 event we place three barriers around the entry price:

- **Upper** barrier:  ``entry · (1 + profit_take_mult · σ_t)``
- **Lower** barrier:  ``entry · (1 - stop_loss_mult · σ_t)``
- **Vertical** barrier: ``vertical_barrier_bars`` bars after the event

where ``σ_t`` is the causal EWM volatility at the event time (Vulnerability 4 of
the AFML audit — handled by ``afml.labeling.volatility.ewm_volatility``'s
``causal_shift=True`` default). **No fixed pips, no fixed prices** — every
barrier self-scales to the local volatility regime.

Intra-bar barrier touch detection (AFML audit Vulnerability 1 — path-dependency):
- Barrier touches are checked against each bar's **high** and **low**, NOT just
  ``close``. A price spike that pierces the stop-loss mid-bar must register as
  a stop hit even if the bar closes back inside the barriers.
- **Conflict-resolution rule:** when a single bar's high ≥ upper AND its low ≤
  lower (a dual-touch wide bar), we conservatively treat the event as a
  stop-loss hit (``label = 0``). Penalizing ambiguity → conservative risk
  modeling.

Labeling (binary meta-label):
- ``y = 1`` if the entry's *intended direction* succeeds:
  - long  + upper hit first
  - short + lower hit first
- ``y = 0`` otherwise (stop hit against the position, dual-touch ambiguity, or
  vertical horizon reached with a non-favorable outcome). Brain 2 will learn to
  predict ``P(y = 1 | features)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
import numpy.typing as npt
import polars as pl

from afml.labeling.volatility import ewm_volatility


@dataclass(frozen=True, slots=True)
class TripleBarrierLabels:
    """Labeled output of ``apply_triple_barrier``.

    The dataframe has columns: ``event_timestamp`` (the entry / ``t0``),
    ``side``, ``entry_price``, ``upper_price``, ``lower_price``,
    ``vertical_timestamp`` (the conservative upper bound on the holding
    period), ``barrier_hit`` (``"upper"|"lower"|"vertical"``),
    ``exit_timestamp`` (the **realized** barrier-touch time — the AFML
    label-resolution moment ``t1``), ``exit_price``, ``return_pct``, ``label``
    (0 or 1).

    **Realized vs vertical ``t1`` (AFML 0-4 integration audit V1).** Downstream
    purging / embargo schemes (``PurgedKFold``, ``PurgedWalkForwardCV``) must
    ingest ``exit_timestamp`` as ``t1``, *not* ``vertical_timestamp``. The
    realized exit is when the label becomes known; purging is tighter than the
    vertical-barrier upper bound and recovers training data without leaking.
    """

    df: pl.DataFrame
    profit_take_mult: float
    stop_loss_mult: float
    vertical_barrier_bars: int
    vol_span: int

    @property
    def n_events(self) -> int:
        return int(self.df.height)

    @property
    def n_positive(self) -> int:
        return int((self.df["label"] == 1).sum())

    @property
    def n_negative(self) -> int:
        return int((self.df["label"] == 0).sum())


@numba.njit(cache=True)
def _label_one(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    event_idx: int,
    side: int,
    upper: float,
    lower: float,
    horizon_end: int,
) -> tuple[int, int, int, float, float]:
    """Simulate forward from ``event_idx`` until a barrier is hit.

    Intra-bar touch detection: ``high[i] ≥ upper`` triggers the upper barrier;
    ``low[i] ≤ lower`` triggers the lower barrier. If a single bar touches BOTH,
    we conservatively treat it as a stop-loss (label = 0) per the AFML audit
    conflict-resolution rule.

    Returns
    -------
    ``(label, barrier_code, exit_idx, return_pct, exit_price)``:
    - ``barrier_code``: 0 = vertical, 1 = upper, 2 = lower
    - ``label``: 1 if outcome matches ``side``, else 0
    - ``exit_price``: the barrier price (upper or lower) at hit, or close at
      vertical — i.e., the price a real fill would have realized.
    """
    entry = close[event_idx]
    end = horizon_end if horizon_end < high.shape[0] else high.shape[0] - 1

    for i in range(event_idx + 1, end + 1):
        upper_touched = high[i] >= upper
        lower_touched = low[i] <= lower

        if upper_touched and lower_touched:
            # Dual touch in a single bar → conservative stop-loss.
            # For long  positions, stop = lower; for short positions, stop = upper.
            if side == 1:
                exit_price = lower
                return 0, 2, i, (exit_price - entry) / entry, exit_price
            else:
                exit_price = upper
                return 0, 1, i, (exit_price - entry) / entry, exit_price

        if upper_touched:
            exit_price = upper
            label = 1 if side == 1 else 0
            return label, 1, i, (exit_price - entry) / entry, exit_price

        if lower_touched:
            exit_price = lower
            label = 1 if side == -1 else 0
            return label, 2, i, (exit_price - entry) / entry, exit_price

    # Vertical barrier — exit at the final close.
    final = close[end]
    return 0, 0, end, (final - entry) / entry, final


def apply_triple_barrier(  # noqa: PLR0915 — single-pass linear bookkeeping; splitting hurts readability
    bars: pl.DataFrame,
    events: pl.DataFrame,
    *,
    vol_span: int = 100,
    profit_take_mult: float = 1.0,
    stop_loss_mult: float = 1.0,
    vertical_barrier_bars: int = 20,
) -> TripleBarrierLabels:
    """Label every event with the Triple-Barrier outcome.

    Parameters
    ----------
    bars : OHLCV DataFrame sorted by ``timestamp`` (output of any Phase 1 bar
        generator). Must include ``timestamp`` and ``close``; ``high`` / ``low``
        are used if present for tighter intra-bar barrier detection (currently
        defaults to ``close`` for both upper and lower paths).
    events : DataFrame with ``timestamp`` (event time, equal to a bar close)
        and ``side`` (``"long"`` / ``"short"``). Produced by a ``PrimaryAlpha``
        plugin.
    vol_span : EWM volatility span for the dynamic barrier scaling.
    profit_take_mult, stop_loss_mult : multipliers on σ_t for upper / lower
        barriers. The Blueprint defaults both to 1.0 (one σ each way); tuning
        is allowed but must be parameter-swept and logged.
    vertical_barrier_bars : maximum holding period (# bars after the event).

    Returns
    -------
    ``TripleBarrierLabels`` wrapping the labeled DataFrame.
    """
    if "timestamp" not in events.columns or "side" not in events.columns:
        raise ValueError("events must have 'timestamp' and 'side' columns")
    if vertical_barrier_bars < 1:
        raise ValueError("vertical_barrier_bars must be >= 1")

    bars_sorted = bars.sort("timestamp")
    bar_ts_dtype = bars_sorted["timestamp"].dtype
    bar_ts = bars_sorted["timestamp"].to_numpy()
    close = bars_sorted["close"].to_numpy().astype(np.float64)
    # AFML audit V1: intra-bar high/low drive barrier-touch detection. If a
    # bar frame lacks H/L (e.g., minimal test fixture) fall back to close — the
    # caller is responsible for supplying genuine OHLC in production.
    high = (
        bars_sorted["high"].to_numpy().astype(np.float64)
        if "high" in bars_sorted.columns
        else close
    )
    low = (
        bars_sorted["low"].to_numpy().astype(np.float64) if "low" in bars_sorted.columns else close
    )
    n = close.shape[0]

    # EWM volatility on log-returns (causal).
    log_close = np.log(close)
    log_returns = np.empty_like(log_close)
    log_returns[0] = np.nan
    log_returns[1:] = np.diff(log_close)
    vol = ewm_volatility(log_returns, span=vol_span)

    # Align event timestamp precision/tz with the bar grid so comparisons hold
    # exactly. Without this an events frame at us-precision and bars at ms-
    # precision would never match even though the calendar instants are equal.
    events_aligned = events.with_columns(pl.col("timestamp").cast(bar_ts_dtype))
    event_ts = events_aligned["timestamp"].to_numpy()
    event_side = events_aligned["side"].to_numpy()
    bar_index_for_ts = np.searchsorted(bar_ts, event_ts)

    event_bar_indices: list[int] = []
    vertical_bar_indices: list[int] = []
    exit_bar_indices: list[int] = []  # AFML 0-4 audit V1: realized t1
    out_side: list[str] = []
    out_entry: list[float] = []
    out_upper: list[float] = []
    out_lower: list[float] = []
    out_barrier: list[str] = []
    out_exit_price: list[float] = []
    out_return: list[float] = []
    out_label: list[int] = []

    for k in range(event_ts.shape[0]):
        idx = int(bar_index_for_ts[k])
        if idx >= n or bar_ts[idx] != event_ts[k]:
            # Event timestamp doesn't exactly align with a bar — skip.
            continue
        sigma = vol[idx]
        if np.isnan(sigma) or sigma <= 0.0:
            continue

        side_str = str(event_side[k])
        side_int = 1 if side_str == "long" else -1

        entry = close[idx]
        upper = entry * (1.0 + profit_take_mult * sigma)
        lower = entry * (1.0 - stop_loss_mult * sigma)
        horizon_end = idx + vertical_barrier_bars

        label, barrier_code, exit_idx, ret_pct, exit_price = _label_one(
            high, low, close, idx, side_int, upper, lower, horizon_end
        )
        barrier_str = {0: "vertical", 1: "upper", 2: "lower"}[barrier_code]

        event_bar_indices.append(idx)
        vertical_bar_indices.append(min(horizon_end, n - 1))
        # AFML 0-4 integration audit V1: capture the *realized* barrier-touch
        # bar index. For upper / lower exits this is < vertical; for vertical
        # exits it equals min(horizon_end, n - 1).
        exit_bar_indices.append(int(exit_idx))
        out_side.append(side_str)
        out_entry.append(float(entry))
        out_upper.append(float(upper))
        out_lower.append(float(lower))
        out_barrier.append(barrier_str)
        out_exit_price.append(float(exit_price))
        out_return.append(float(ret_pct))
        out_label.append(int(label))

    # Build timestamp columns by indexing the original numpy datetime64 array;
    # this gives Polars a proper datetime dtype to consume (a Python list of
    # np.datetime64 scalars is interpreted as Object and fails to cast).
    if event_bar_indices:
        event_ts_arr = bar_ts[np.asarray(event_bar_indices, dtype=np.int64)]
        vertical_ts_arr = bar_ts[np.asarray(vertical_bar_indices, dtype=np.int64)]
        exit_ts_arr = bar_ts[np.asarray(exit_bar_indices, dtype=np.int64)]
    else:
        empty_ts = np.array([], dtype=bar_ts.dtype)
        event_ts_arr = empty_ts
        vertical_ts_arr = empty_ts
        exit_ts_arr = empty_ts

    df = pl.DataFrame({
        "event_timestamp": event_ts_arr,
        "side": out_side,
        "entry_price": out_entry,
        "upper_price": out_upper,
        "lower_price": out_lower,
        "vertical_timestamp": vertical_ts_arr,
        "barrier_hit": out_barrier,
        # AFML 0-4 integration audit V1: realized t1 for downstream purging.
        # ``exit_timestamp ≤ vertical_timestamp`` always; equality holds iff
        # the bar hit the vertical barrier.
        "exit_timestamp": exit_ts_arr,
        "exit_price": out_exit_price,
        "return_pct": out_return,
        "label": out_label,
    })
    return TripleBarrierLabels(
        df=df,
        profit_take_mult=profit_take_mult,
        stop_loss_mult=stop_loss_mult,
        vertical_barrier_bars=vertical_barrier_bars,
        vol_span=vol_span,
    )
