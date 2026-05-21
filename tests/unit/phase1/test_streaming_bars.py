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

from afml.data.bars.streaming import (
    build_information_bars_streaming,
    calibrate_information_bar_threshold,
)
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


# --- fixed-threshold (calibrated) mode — Stage-1 EWMA-collapse fix --------------


@pytest.mark.parametrize("bar_type", ["tick_imbalance", "tick_run"])
@pytest.mark.parametrize("chunk_rows", [137, 1000, 100_000])
def test_fixed_threshold_streaming_matches_single_pass(bar_type: str, chunk_rows: int) -> None:
    """The fixed-threshold path must also be bar-for-bar identical across chunks."""
    ticks = _synth_ticks()
    single = (
        build_tick_imbalance_bars(ticks, fixed_threshold=20.0)
        if bar_type == "tick_imbalance"
        else build_tick_run_bars(ticks, fixed_threshold=20.0)
    )
    stream = build_information_bars_streaming(
        ticks.lazy(), bar_type=bar_type, fixed_threshold=20.0, chunk_rows=chunk_rows
    )
    _assert_bars_identical(single, stream)


@pytest.mark.parametrize("bar_type", ["tick_imbalance", "tick_run"])
def test_calibration_hits_target_ticks_per_bar(bar_type: str) -> None:
    """Calibrated fixed threshold lands the realised ticks/bar near the target."""
    ticks = _synth_ticks(n=40000, seed=5)
    target = 250.0
    h = calibrate_information_bar_threshold(ticks, bar_type=bar_type, target_ticks_per_bar=target)
    assert h > 0.0
    bars = build_information_bars_streaming(
        ticks.lazy(), bar_type=bar_type, fixed_threshold=h, chunk_rows=10_000
    )
    mean_ticks = float(bars["n_ticks"].to_numpy().mean())
    assert 0.5 * target <= mean_ticks <= 2.0 * target  # within a factor of 2 of the target


def test_fixed_threshold_gives_controlled_bar_count() -> None:
    """A fixed threshold yields a controlled ticks/bar — never the ~1-tick collapse.

    (The adaptive-EWMA runaway-collapse is shown on real 2-year data, where it
    produced 17.6M bars from 49.5M ticks; here we pin that the fixed path stays
    coarse and bounded.)
    """
    ticks = _synth_ticks(n=30000, seed=2)
    trb = build_tick_run_bars(ticks, fixed_threshold=100.0)
    tib = build_tick_imbalance_bars(ticks, fixed_threshold=100.0)
    trb_tpb = 30000 / max(trb.height, 1)
    tib_tpb = 30000 / max(tib.height, 1)
    # TRB closes at max(n±)≥100 → ~100-300 ticks/bar; never the ~1-tick collapse.
    assert trb_tpb >= 50.0
    assert tib_tpb >= 50.0
