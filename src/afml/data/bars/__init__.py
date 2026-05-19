"""Information-driven bar generators (Blueprint §3.1).

Three bar types are implemented:

- ``time_bars``        — clock-time sampling (baseline; non-informational).
- ``tick_imbalance``   — TIB: sample when |Σ tick-sign| exceeds an EWM-adaptive
                         threshold (López de Prado 2018, Ch. 2).
- ``tick_run``         — TRB: sample when max of consecutive same-side tick runs
                         crosses an EWM-adaptive threshold.

Tick-based bars produce more normally-distributed returns than time bars in
volatile/non-stationary markets; the bar-type selector picks whichever yields the
lowest Jarque-Bera statistic on the resulting returns.
"""

from afml.data.bars.tick_imbalance import build_tick_imbalance_bars
from afml.data.bars.tick_run import build_tick_run_bars
from afml.data.bars.time_bars import build_time_bars

__all__ = [
    "build_tick_imbalance_bars",
    "build_tick_run_bars",
    "build_time_bars",
]
