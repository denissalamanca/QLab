"""Bollinger Band Mean-Reversion plugin (Brain 1).

Best in ranging regimes. The plugin fires when price crosses **outside** the
SMA ± k·σ Bollinger band, betting on reversion back to the mean:

- Cross BELOW the lower band → ``side = "long"``  (price oversold → bounce up).
- Cross ABOVE the upper band → ``side = "short"`` (price overbought → revert down).

All rolling statistics are strictly causal (``.shift(1)`` on the band so the
crossing decision at index ``t`` uses bands computed from prior bars only).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from afml.labeling.primary_alphas.base import PrimaryAlpha, register_alpha


@register_alpha
class BollingerMeanReversion(PrimaryAlpha):
    """Mean-reversion entry when close crosses outside the SMA ± k·σ band."""

    algorithmic_family = "bbands_meanrev"

    def __init__(self, *, sma_span: int = 20, num_std: float = 2.0) -> None:
        super().__init__(sma_span=sma_span, num_std=num_std)
        self.sma_span = sma_span
        self.num_std = num_std

    def detect(self, bars: pl.DataFrame) -> pl.DataFrame:
        if bars.height < self.sma_span + 2:
            return _empty_events()

        close = bars["close"].to_numpy().astype(np.float64)
        n = close.size

        # Causal rolling mean + std using the prior ``sma_span`` bars only.
        # rolling().shift(1) so that band at index t depends on close[0..t-1].
        sma = (
            pl
            .Series(close)
            .rolling_mean(window_size=self.sma_span, min_samples=self.sma_span)
            .shift(1)
            .to_numpy()
        )
        std = (
            pl
            .Series(close)
            .rolling_std(window_size=self.sma_span, min_samples=self.sma_span)
            .shift(1)
            .to_numpy()
        )

        upper = sma + self.num_std * std
        lower = sma - self.num_std * std

        events_idx: list[int] = []
        events_side: list[str] = []
        for i in range(1, n):
            if np.isnan(upper[i]) or np.isnan(lower[i]):
                continue
            # Compare current and previous close against the (causal) band.
            prev_above_upper = close[i - 1] > upper[i - 1] if not np.isnan(upper[i - 1]) else False
            prev_below_lower = close[i - 1] < lower[i - 1] if not np.isnan(lower[i - 1]) else False
            curr_above_upper = close[i] > upper[i]
            curr_below_lower = close[i] < lower[i]

            # Mean-reversion: take a long when price has crossed BELOW the lower
            # band (oversold); take a short when it has crossed ABOVE the upper
            # band (overbought). Only fire on the crossing, not while outside.
            if curr_below_lower and not prev_below_lower:
                events_idx.append(i)
                events_side.append("long")
            elif curr_above_upper and not prev_above_upper:
                events_idx.append(i)
                events_side.append("short")

        if not events_idx:
            return _empty_events()

        timestamps = bars["timestamp"].to_numpy()[events_idx]
        return pl.DataFrame({"timestamp": timestamps, "side": events_side})


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8})
