"""Phase 1 — Time bar aggregation."""

from __future__ import annotations

import polars as pl
import pytest

from afml.data.bars.time_bars import build_time_bars


@pytest.mark.phase1
def test_time_bars_minimum_columns(tick_stream_medium: pl.DataFrame) -> None:
    bars = build_time_bars(tick_stream_medium, interval="1m")
    expected = {"timestamp", "open", "high", "low", "close", "volume", "n_ticks"}
    assert expected.issubset(set(bars.columns))


@pytest.mark.phase1
def test_time_bars_high_ge_low(tick_stream_medium: pl.DataFrame) -> None:
    bars = build_time_bars(tick_stream_medium, interval="1m")
    assert bool((bars["high"] >= bars["low"]).all())
    assert bool((bars["high"] >= bars["open"]).all())
    assert bool((bars["high"] >= bars["close"]).all())
    assert bool((bars["low"] <= bars["open"]).all())
    assert bool((bars["low"] <= bars["close"]).all())


@pytest.mark.phase1
def test_time_bars_n_ticks_positive(tick_stream_medium: pl.DataFrame) -> None:
    bars = build_time_bars(tick_stream_medium, interval="1m")
    assert bool((bars["n_ticks"] > 0).all())


@pytest.mark.phase1
def test_time_bars_volume_is_sum_of_inputs(tick_stream_medium: pl.DataFrame) -> None:
    bars = build_time_bars(tick_stream_medium, interval="1m")
    total = float(bars["volume"].sum())
    expected = float((tick_stream_medium["bid_volume"] + tick_stream_medium["ask_volume"]).sum())
    assert total == pytest.approx(expected, rel=1e-9)


@pytest.mark.phase1
def test_time_bars_count_matches_minute_buckets(tick_stream_medium: pl.DataFrame) -> None:
    """3000 ticks × 100 ms = 300 s = 5 min → 5 one-minute bars (give or take 1 at edges)."""
    bars = build_time_bars(tick_stream_medium, interval="1m")
    assert 4 <= bars.height <= 6


@pytest.mark.phase1
def test_time_bars_sorted_by_timestamp(tick_stream_medium: pl.DataFrame) -> None:
    bars = build_time_bars(tick_stream_medium, interval="1m")
    ts = bars["timestamp"].to_list()
    assert ts == sorted(ts)
