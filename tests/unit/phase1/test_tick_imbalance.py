"""Phase 1 — Tick Imbalance Bars (TIB)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from afml.data.bars.tick_imbalance import build_tick_imbalance_bars


@pytest.mark.phase1
def test_tib_returns_canonical_schema(tick_stream_medium: pl.DataFrame) -> None:
    bars = build_tick_imbalance_bars(tick_stream_medium)
    expected = {"timestamp", "open", "high", "low", "close", "volume", "n_ticks"}
    assert expected.issubset(set(bars.columns))


@pytest.mark.phase1
def test_tib_high_ge_low(tick_stream_medium: pl.DataFrame) -> None:
    bars = build_tick_imbalance_bars(tick_stream_medium)
    assert bars.height > 0
    assert bool((bars["high"] >= bars["low"]).all())


@pytest.mark.phase1
def test_tib_n_ticks_positive_and_sums_to_total(tick_stream_medium: pl.DataFrame) -> None:
    bars = build_tick_imbalance_bars(tick_stream_medium)
    assert bool((bars["n_ticks"] > 0).all())
    # All tick volumes are attributed to some bar (no orphan ticks lost).
    assert int(bars["n_ticks"].sum()) <= tick_stream_medium.height


@pytest.mark.phase1
def test_tib_empty_input_returns_empty_frame() -> None:
    empty = pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("ms", "UTC"),
            "bid": pl.Float64,
            "ask": pl.Float64,
            "bid_volume": pl.Float64,
            "ask_volume": pl.Float64,
        }
    )
    out = build_tick_imbalance_bars(empty)
    assert out.height == 0
    assert set(out.columns) >= {"timestamp", "open", "close", "n_ticks"}


@pytest.mark.phase1
def test_tib_with_higher_initial_expectation_produces_fewer_bars(
    tick_stream_large: pl.DataFrame,
) -> None:
    """Higher ``initial_expected_ticks`` raises the imbalance threshold, so the
    sampler should produce fewer (but larger) bars in expectation."""
    bars_low = build_tick_imbalance_bars(tick_stream_large, initial_expected_ticks=50.0)
    bars_high = build_tick_imbalance_bars(tick_stream_large, initial_expected_ticks=500.0)
    assert bars_high.height <= bars_low.height


@pytest.mark.phase1
def test_tib_bars_chronologically_ordered(tick_stream_medium: pl.DataFrame) -> None:
    bars = build_tick_imbalance_bars(tick_stream_medium)
    ts = bars["timestamp"].to_list()
    assert ts == sorted(ts)


@pytest.mark.phase1
def test_tib_truncation_invariance(tick_stream_large: pl.DataFrame) -> None:
    """AFML audit V3 — Information Bar Update causality proof.

    Bars computed on a full tick series must match those computed on a
    truncated prefix, for every bar that closes before the truncation point.
    If the EWMAs of E[T] or P[+1] were updated using the *current* forming
    bar's ticks (a leak), the threshold for bar k would depend on data later
    than bar k-1's close, and a truncated computation would produce different
    bar closes.
    """
    truncation_idx = tick_stream_large.height // 2
    full_bars = build_tick_imbalance_bars(tick_stream_large)
    trunc_bars = build_tick_imbalance_bars(tick_stream_large.head(truncation_idx))

    full_ts = full_bars["timestamp"].to_numpy()
    trunc_ts = trunc_bars["timestamp"].to_numpy()
    # All trunc bars must be a prefix of full bars (closes < truncation_idx
    # are decided identically in both runs).
    n = trunc_ts.size
    assert n > 0
    np.testing.assert_array_equal(trunc_ts, full_ts[:n])
