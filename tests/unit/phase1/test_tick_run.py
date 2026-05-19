"""Phase 1 — Tick Run Bars (TRB)."""

from __future__ import annotations

import polars as pl
import pytest

from afml.data.bars.tick_run import build_tick_run_bars


@pytest.mark.phase1
def test_trb_returns_canonical_schema(tick_stream_medium: pl.DataFrame) -> None:
    bars = build_tick_run_bars(tick_stream_medium)
    expected = {"timestamp", "open", "high", "low", "close", "volume", "n_ticks"}
    assert expected.issubset(set(bars.columns))


@pytest.mark.phase1
def test_trb_high_ge_low(tick_stream_medium: pl.DataFrame) -> None:
    bars = build_tick_run_bars(tick_stream_medium)
    assert bars.height > 0
    assert bool((bars["high"] >= bars["low"]).all())


@pytest.mark.phase1
def test_trb_higher_expectation_yields_fewer_bars(tick_stream_large: pl.DataFrame) -> None:
    bars_low = build_tick_run_bars(tick_stream_large, initial_expected_ticks=50.0)
    bars_high = build_tick_run_bars(tick_stream_large, initial_expected_ticks=500.0)
    assert bars_high.height <= bars_low.height


@pytest.mark.phase1
def test_trb_empty_input_returns_empty_frame() -> None:
    empty = pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("ms", "UTC"),
            "bid": pl.Float64,
            "ask": pl.Float64,
            "bid_volume": pl.Float64,
            "ask_volume": pl.Float64,
        }
    )
    out = build_tick_run_bars(empty)
    assert out.height == 0


@pytest.mark.phase1
def test_trb_total_attributed_ticks_le_input(tick_stream_medium: pl.DataFrame) -> None:
    bars = build_tick_run_bars(tick_stream_medium)
    assert int(bars["n_ticks"].sum()) <= tick_stream_medium.height
