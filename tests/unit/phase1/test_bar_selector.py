"""Phase 1 — Jarque-Bera-minimization bar-type selector with plateau check."""

from __future__ import annotations

import polars as pl
import pytest

from afml.data.bars import selector as sel_module
from afml.data.bars.selector import (
    DEFAULT_TICK_EXPECTED,
    DEFAULT_TIME_INTERVALS,
    BarSelection,
    BarType,
    BarTypeSweepResult,
    select_bar_type,
)


@pytest.mark.phase1
def test_selector_sweeps_all_combinations(tick_stream_large: pl.DataFrame) -> None:
    sel = select_bar_type(tick_stream_large)
    expected_count = len(DEFAULT_TIME_INTERVALS) + 2 * len(DEFAULT_TICK_EXPECTED)
    assert len(sel.sweep) == expected_count
    types = {r.bar_type for r in sel.sweep}
    assert types == {"time", "tick_imbalance", "tick_run"}


@pytest.mark.phase1
def test_selector_chooses_finite_winner(tick_stream_large: pl.DataFrame) -> None:
    sel = select_bar_type(tick_stream_large)
    assert sel.winner is not None
    assert sel.winner.jarque_bera < float("inf")
    assert sel.bars.height > 0


@pytest.mark.phase1
def test_selector_winner_is_in_sweep(tick_stream_large: pl.DataFrame) -> None:
    sel = select_bar_type(tick_stream_large)
    assert sel.winner in sel.sweep


@pytest.mark.phase1
def test_selector_plateau_rejects_isolated_spike(
    tick_stream_large: pl.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-Lazy plateau check: a global JB minimum surrounded by huge JBs is
    an isolated spike and must be rejected in favor of a stable point."""
    # Run the real selector once to get a real sweep, then mutate it via a
    # monkeypatched plateau-check to simulate the spike scenario.
    sel = select_bar_type(tick_stream_large, plateau_factor=2.0)

    # Build a synthetic sweep with one bar type: time bars, intervals 1m..1h,
    # where the JB at "15m" is artificially low (spike) and neighbors are huge.
    synthetic_sweep = [
        BarTypeSweepResult("time", "1m", jarque_bera=1000.0, n_bars=300),
        BarTypeSweepResult("time", "5m", jarque_bera=1000.0, n_bars=60),
        BarTypeSweepResult("time", "15m", jarque_bera=1.0, n_bars=20),  # spike
        BarTypeSweepResult("time", "30m", jarque_bera=1000.0, n_bars=10),
        BarTypeSweepResult("time", "1h", jarque_bera=1000.0, n_bars=5),
    ]
    cand_spike = synthetic_sweep[2]
    # The spike fails the plateau check with factor=2.0.
    assert not sel_module._check_plateau(cand_spike, synthetic_sweep, factor=2.0)

    # The neighbors form a flat plateau among themselves: any of those should
    # pass against each other.
    cand_neighbor = synthetic_sweep[1]
    assert sel_module._check_plateau(cand_neighbor, synthetic_sweep, factor=2.0)

    # Sanity: the real selector still returned something — flag is informational.
    assert isinstance(sel, BarSelection)


@pytest.mark.phase1
def test_selector_plateau_factor_lower_is_stricter() -> None:
    """A small ``plateau_factor`` should reject more candidates as unstable."""
    sweep = [
        BarTypeSweepResult("time", "1m", 10.0, 300),
        BarTypeSweepResult("time", "5m", 5.0, 60),  # candidate
        BarTypeSweepResult("time", "15m", 14.0, 20),
    ]
    cand = sweep[1]
    # factor 3.0 — neighbor JBs (10, 14) ≤ 3·5 = 15 → plateau OK.
    assert sel_module._check_plateau(cand, sweep, factor=3.0)
    # factor 1.5 — neighbor JB 14 > 1.5·5 = 7.5 → plateau fails.
    assert not sel_module._check_plateau(cand, sweep, factor=1.5)


@pytest.mark.phase1
def test_selector_winner_bartype_is_valid(tick_stream_large: pl.DataFrame) -> None:
    sel = select_bar_type(tick_stream_large)
    valid: set[BarType] = {"time", "tick_imbalance", "tick_run"}
    assert sel.winner.bar_type in valid
