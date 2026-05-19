"""Information-bar type selector via Jarque-Bera minimization (Blueprint §3.1).

Picks the bar type + scale parameter whose returns distribution is closest to
normal (lowest Jarque-Bera statistic). Anti-Lazy: the full parameter sweep is
returned so the choice is auditable, and the winner must sit on a **stable
plateau** — adjacent parameter values must produce JB statistics within a
configurable factor of the minimum. Isolated spikes are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl

from afml.data.bars.tick_imbalance import build_tick_imbalance_bars
from afml.data.bars.tick_run import build_tick_run_bars
from afml.data.bars.time_bars import build_time_bars
from afml.data.stationarity import jarque_bera_statistic

BarType = Literal["time", "tick_imbalance", "tick_run"]


@dataclass(frozen=True, slots=True)
class BarTypeSweepResult:
    """One point in the bar-type × parameter sweep."""

    bar_type: BarType
    parameter: float | str  # interval string for time, float for TIB/TRB
    jarque_bera: float
    n_bars: int

    @property
    def parameter_label(self) -> str:
        return str(self.parameter)


@dataclass(frozen=True, slots=True)
class BarSelection:
    """Outcome of ``select_bar_type``."""

    winner: BarTypeSweepResult
    bars: pl.DataFrame
    sweep: list[BarTypeSweepResult]
    plateau_factor_used: float
    stable: bool = field(default=True)


DEFAULT_TIME_INTERVALS: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h")
DEFAULT_TICK_EXPECTED: tuple[float, ...] = (50.0, 100.0, 200.0, 500.0, 1000.0)


def _log_returns(bars: pl.DataFrame) -> np.ndarray:
    closes = bars["close"].to_numpy().astype(np.float64)
    if closes.size < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(np.log(closes))


def _score(bars: pl.DataFrame) -> tuple[float, int]:
    rets = _log_returns(bars)
    if rets.size < 30:
        return float("inf"), int(bars.height)
    try:
        return jarque_bera_statistic(rets), int(bars.height)
    except ValueError:
        return float("inf"), int(bars.height)


def _check_plateau(
    candidate: BarTypeSweepResult,
    sweep: list[BarTypeSweepResult],
    factor: float,
) -> bool:
    """A candidate is on a stable plateau iff every adjacent point of the SAME
    bar type has ``JB ≤ factor · candidate.JB``.

    "Adjacent" = neighboring index after sorting that bar type's results by
    parameter value (ascending). Edge points have only one neighbor; that one
    must still satisfy the threshold.
    """
    same_type = sorted(
        (s for s in sweep if s.bar_type == candidate.bar_type),
        key=lambda s: (
            float(s.parameter)
            if isinstance(s.parameter, (int, float))
            else _interval_seconds(str(s.parameter))
        ),
    )
    idx = same_type.index(candidate)
    neighbors: list[BarTypeSweepResult] = []
    if idx > 0:
        neighbors.append(same_type[idx - 1])
    if idx < len(same_type) - 1:
        neighbors.append(same_type[idx + 1])
    if not neighbors:
        return True  # singleton sweep — no plateau to verify
    bound = candidate.jarque_bera * factor
    return all(np.isfinite(n.jarque_bera) and n.jarque_bera <= bound for n in neighbors)


def _interval_seconds(s: str) -> float:
    """Tiny parser for Polars duration strings used for sorting only."""
    unit = s[-1]
    val = float(s[:-1])
    multipliers = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    return val * multipliers.get(unit, 1.0)


def select_bar_type(
    ticks: pl.DataFrame,
    *,
    time_intervals: tuple[str, ...] = DEFAULT_TIME_INTERVALS,
    tick_expected_ticks: tuple[float, ...] = DEFAULT_TICK_EXPECTED,
    plateau_factor: float = 2.0,
) -> BarSelection:
    """Run the bar-type tournament and return the JB-minimizing, plateau-stable winner.

    Anti-Lazy: ``BarSelection.sweep`` is the full audit trail. If the global JB
    minimum is on an isolated spike (neighbors > ``plateau_factor × JB_min``),
    the selector walks down the sweep until it finds a stable winner.

    Parameters
    ----------
    ticks : tick DataFrame (``timestamp, bid, ask, bid_volume, ask_volume``).
    time_intervals : Polars duration strings to try for time bars.
    tick_expected_ticks : warm-up E[T] values to try for TIB and TRB.
    plateau_factor : neighbors' JB must satisfy ``JB_neighbor ≤ factor · JB_min``.

    Returns
    -------
    ``BarSelection`` with the winning bars, the winning sweep entry, the full
    sweep log, and the plateau diagnostic.
    """
    sweep: list[BarTypeSweepResult] = []
    bars_by_key: dict[tuple[BarType, float | str], pl.DataFrame] = {}

    for interval in time_intervals:
        bars = build_time_bars(ticks, interval=interval)
        jb, n = _score(bars)
        entry = BarTypeSweepResult("time", interval, jb, n)
        sweep.append(entry)
        bars_by_key[("time", interval)] = bars

    for et in tick_expected_ticks:
        bars = build_tick_imbalance_bars(ticks, initial_expected_ticks=et)
        jb, n = _score(bars)
        entry = BarTypeSweepResult("tick_imbalance", et, jb, n)
        sweep.append(entry)
        bars_by_key[("tick_imbalance", et)] = bars

    for et in tick_expected_ticks:
        bars = build_tick_run_bars(ticks, initial_expected_ticks=et)
        jb, n = _score(bars)
        entry = BarTypeSweepResult("tick_run", et, jb, n)
        sweep.append(entry)
        bars_by_key[("tick_run", et)] = bars

    # Sort by JB ascending; walk down until we find a stable-plateau candidate.
    ordered = sorted(sweep, key=lambda r: (np.isinf(r.jarque_bera), r.jarque_bera))
    winner: BarTypeSweepResult | None = None
    stable = True
    for cand in ordered:
        if not np.isfinite(cand.jarque_bera):
            continue
        if _check_plateau(cand, sweep, plateau_factor):
            winner = cand
            break
    if winner is None:
        # No stable plateau anywhere — fall back to the absolute minimum and flag.
        winner = next(c for c in ordered if np.isfinite(c.jarque_bera))
        stable = False

    return BarSelection(
        winner=winner,
        bars=bars_by_key[(winner.bar_type, winner.parameter)],
        sweep=sweep,
        plateau_factor_used=plateau_factor,
        stable=stable,
    )
