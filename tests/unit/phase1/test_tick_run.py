"""Phase 1 — Tick Run Bars (TRB)."""

from __future__ import annotations

import numpy as np
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


@pytest.mark.phase1
def test_trb_truncation_invariance(tick_stream_large: pl.DataFrame) -> None:
    """AFML audit V3 — TRB shares TIB's causal-update invariant.

    See ``test_tib_truncation_invariance`` for the rationale: bar closes in
    the prefix [0, truncation_idx) must be identical between full-series and
    truncated-series runs.
    """
    truncation_idx = tick_stream_large.height // 2
    full_bars = build_tick_run_bars(tick_stream_large)
    trunc_bars = build_tick_run_bars(tick_stream_large.head(truncation_idx))

    full_ts = full_bars["timestamp"].to_numpy()
    trunc_ts = trunc_bars["timestamp"].to_numpy()
    n = trunc_ts.size
    assert n > 0
    np.testing.assert_array_equal(trunc_ts, full_ts[:n])
