"""Streaming bar construction == single-pass (Ops M1 memory-hardening).

The resumable row-slice kernels in :mod:`afml.data.bars.streaming` exist so a
360M-tick asset can be barred without OOM. Their correctness contract is
**bar-for-bar identity** with the single-pass builders, across every slice
boundary. These tests pin that contract (the truncation-hash invariant's
streaming cousin) so the JB bar-type tournament is provably unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from afml.data.bars.streaming import build_information_bars_streaming
from afml.data.bars.tick_imbalance import build_tick_imbalance_bars
from afml.data.bars.tick_run import build_tick_run_bars
from afml.data.bars.time_bars import build_time_bars

pytestmark = pytest.mark.phase1

_OHLCV = ("open", "high", "low", "close", "volume", "n_ticks")


def _synth_ticks(n: int = 6000, seed: int = 11) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    ts = [t0 + timedelta(seconds=i) for i in range(n)]
    mid = 100.0 + np.cumsum(rng.standard_normal(n) * 0.01)
    mid[::40] = mid[np.maximum(np.arange(n)[::40] - 1, 0)]  # equal-price ticks (sign carry)
    return pl.DataFrame(
        {
            "timestamp": ts,
            "bid": mid - 0.005,
            "ask": mid + 0.005,
            "bid_volume": rng.integers(1, 10, n).astype(np.int64),
            "ask_volume": rng.integers(1, 10, n).astype(np.int64),
        },
        schema={
            "timestamp": pl.Datetime("ms", "UTC"),
            "bid": pl.Float64,
            "ask": pl.Float64,
            "bid_volume": pl.Int64,
            "ask_volume": pl.Int64,
        },
    )


def _assert_bars_identical(single: pl.DataFrame, stream: pl.DataFrame) -> None:
    a = single.with_columns(pl.col("timestamp").cast(pl.Int64))
    b = stream.with_columns(pl.col("timestamp").cast(pl.Int64))
    assert a.shape == b.shape, f"row count differs: {a.shape} vs {b.shape}"
    np.testing.assert_array_equal(a["timestamp"].to_numpy(), b["timestamp"].to_numpy())
    for col in _OHLCV:
        np.testing.assert_array_equal(
            a[col].to_numpy(), b[col].to_numpy(), err_msg=f"column {col} diverged"
        )


@pytest.mark.parametrize("expected_ticks", [30.0, 100.0])
@pytest.mark.parametrize("chunk_rows", [137, 1000, 100_000])
def test_tib_streaming_matches_single_pass(expected_ticks: float, chunk_rows: int) -> None:
    ticks = _synth_ticks()
    single = build_tick_imbalance_bars(ticks, initial_expected_ticks=expected_ticks)
    stream = build_information_bars_streaming(
        ticks.lazy(),
        bar_type="tick_imbalance",
        initial_expected_ticks=expected_ticks,
        chunk_rows=chunk_rows,
    )
    _assert_bars_identical(single, stream)


@pytest.mark.parametrize("expected_ticks", [30.0, 100.0])
@pytest.mark.parametrize("chunk_rows", [137, 1000, 100_000])
def test_trb_streaming_matches_single_pass(expected_ticks: float, chunk_rows: int) -> None:
    ticks = _synth_ticks()
    single = build_tick_run_bars(ticks, initial_expected_ticks=expected_ticks)
    stream = build_information_bars_streaming(
        ticks.lazy(),
        bar_type="tick_run",
        initial_expected_ticks=expected_ticks,
        chunk_rows=chunk_rows,
    )
    _assert_bars_identical(single, stream)


def test_time_bars_streaming_matches_eager() -> None:
    ticks = _synth_ticks()
    eager = build_time_bars(ticks, interval="90s")
    stream = build_time_bars(ticks.lazy(), interval="90s", streaming=True)
    _assert_bars_identical(eager, stream)


def test_streaming_rejects_unknown_bar_type() -> None:
    with pytest.raises(ValueError, match=r"tick_imbalance\|tick_run"):
        build_information_bars_streaming(_synth_ticks().lazy(), bar_type="volume")


def test_streaming_empty_input_returns_empty_frame() -> None:
    empty = _synth_ticks(n=0)
    out = build_information_bars_streaming(empty.lazy(), bar_type="tick_imbalance")
    assert out.height == 0
    assert out.columns == ["timestamp", "open", "high", "low", "close", "volume", "n_ticks"]
