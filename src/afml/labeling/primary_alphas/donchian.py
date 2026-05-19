"""Donchian Channel Breakout plugin (Brain 1).

Best in trending / momentum regimes. The plugin fires when the close breaks
**outside** the highest-high / lowest-low envelope of the prior ``window`` bars,
betting on continuation:

- Close > prior ``window`` max  →  ``side = "long"``   (upside breakout).
- Close < prior ``window`` min  →  ``side = "short"``  (downside breakout).

All rolling extremes are strictly causal: the channel at index ``t`` is computed
from bars at indices ``[t - window, t - 1]``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from afml.labeling.primary_alphas.base import PrimaryAlpha, register_alpha


@register_alpha
class DonchianBreakout(PrimaryAlpha):
    """Donchian channel breakout entry on bar closes."""

    algorithmic_family = "donchian_breakout"

    def __init__(self, *, window: int = 20) -> None:
        super().__init__(window=window)
        self.window = window

    def detect(self, bars: pl.DataFrame) -> pl.DataFrame:
        if bars.height < self.window + 2:
            return _empty_events()

        close = bars["close"].to_numpy().astype(np.float64)
        high = bars["high"].to_numpy().astype(np.float64)
        low = bars["low"].to_numpy().astype(np.float64)
        n = close.size

        # Causal rolling max / min using only the prior ``window`` bars.
        upper = (
            pl
            .Series(high)
            .rolling_max(window_size=self.window, min_samples=self.window)
            .shift(1)
            .to_numpy()
        )
        lower = (
            pl
            .Series(low)
            .rolling_min(window_size=self.window, min_samples=self.window)
            .shift(1)
            .to_numpy()
        )

        events_idx: list[int] = []
        events_side: list[str] = []
        in_long = False
        in_short = False
        for i in range(self.window, n):
            if np.isnan(upper[i]) or np.isnan(lower[i]):
                continue
            if close[i] > upper[i]:
                if not in_long:
                    events_idx.append(i)
                    events_side.append("long")
                    in_long = True
                    in_short = False
            elif close[i] < lower[i]:
                if not in_short:
                    events_idx.append(i)
                    events_side.append("short")
                    in_short = True
                    in_long = False
            else:
                # Back inside the channel → reset breakout state so the next
                # breach generates a new event.
                in_long = False
                in_short = False

        if not events_idx:
            return _empty_events()

        timestamps = bars["timestamp"].to_numpy()[events_idx]
        return pl.DataFrame({"timestamp": timestamps, "side": events_side})


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8})
