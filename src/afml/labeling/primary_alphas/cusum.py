"""Symmetric CUSUM event filter (Blueprint §4.1).

A structural volatility-event detector. Cumulative log-return is tracked in two
running statistics:

    S⁺_t = max(0, S⁺_{t-1} + r_t)
    S⁻_t = min(0, S⁻_{t-1} + r_t)

An event is sampled when either crosses the (causal) threshold ``h_t`` derived
from the EWM rolling volatility. Both running sums are reset to zero on each
event so the next event requires a fresh accumulation. ``threshold_multiplier``
scales the EWM vol; per the AFML anti-bias rule the threshold itself is
data-driven, not hard-coded.

Side convention: ``"long"`` when ``S⁺`` breaches (continuation up), ``"short"``
when ``S⁻`` breaches (continuation down). Mean-reversion and breakout
interpretations are handled by the Bollinger and Donchian plugins.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import polars as pl

from afml.labeling.primary_alphas.base import PrimaryAlpha, register_alpha
from afml.labeling.volatility import ewm_volatility


@numba.njit(cache=True)
def _cusum_breach_indices(
    returns: npt.NDArray[np.float64],
    threshold: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int8]]:
    """Walk the symmetric CUSUM accumulators and return breach indices + sides."""
    n = returns.shape[0]
    out_idx = np.empty(n, dtype=np.int64)
    out_side = np.empty(n, dtype=np.int8)
    count = 0
    s_pos = 0.0
    s_neg = 0.0
    for i in range(n):
        r = returns[i]
        th = threshold[i]
        if np.isnan(r) or np.isnan(th) or th <= 0.0:
            # Threshold must be strictly positive — otherwise a flat (zero-vol)
            # series would trigger a false breach on the very first bar.
            continue
        s_pos_new = s_pos + r
        s_pos_new = max(s_pos_new, 0.0)
        s_neg_new = s_neg + r
        s_neg_new = min(s_neg_new, 0.0)
        if s_pos_new >= th:
            out_idx[count] = i
            out_side[count] = 1
            count += 1
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg_new <= -th:
            out_idx[count] = i
            out_side[count] = -1
            count += 1
            s_pos = 0.0
            s_neg = 0.0
        else:
            s_pos = s_pos_new
            s_neg = s_neg_new
    return out_idx[:count], out_side[:count]


@register_alpha
class SymmetricCUSUM(PrimaryAlpha):
    """Symmetric CUSUM filter on bar log-returns with EWM-vol threshold."""

    algorithmic_family = "cusum"

    def __init__(
        self,
        *,
        vol_span: int = 100,
        threshold_multiplier: float = 1.0,
    ) -> None:
        super().__init__(vol_span=vol_span, threshold_multiplier=threshold_multiplier)
        self.vol_span = vol_span
        self.threshold_multiplier = threshold_multiplier

    def detect(self, bars: pl.DataFrame) -> pl.DataFrame:
        close = bars["close"].to_numpy().astype(np.float64)
        if close.size < 2:
            return _empty_events()

        log_close = np.log(close)
        log_returns = np.empty_like(log_close)
        log_returns[0] = np.nan
        log_returns[1:] = np.diff(log_close)

        vol = ewm_volatility(log_returns, span=self.vol_span)
        threshold = vol * self.threshold_multiplier

        idx, side = _cusum_breach_indices(log_returns, threshold)
        if idx.size == 0:
            return _empty_events()

        timestamps = bars["timestamp"].to_numpy()[idx]
        side_str = np.where(side == 1, "long", "short")
        return pl.DataFrame({"timestamp": timestamps, "side": side_str.tolist()})


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8})
