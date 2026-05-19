"""Phase 1 test fixtures — synthetic tick streams."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest


def _build_tick_frame(
    n_ticks: int,
    *,
    base_price: float = 100.0,
    drift: float = 0.0,
    vol: float = 0.001,
    spread: float = 0.0002,
    tick_ms: int = 100,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate a synthetic random-walk tick stream in the AFML schema."""
    rng = np.random.default_rng(seed)
    increments = rng.standard_normal(n_ticks) * vol + drift
    mid = base_price + np.cumsum(increments)
    half = spread / 2.0
    bid = mid - half
    ask = mid + half
    bid_vol = rng.uniform(0.1, 5.0, n_ticks)
    ask_vol = rng.uniform(0.1, 5.0, n_ticks)
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [t0 + timedelta(milliseconds=i * tick_ms) for i in range(n_ticks)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "bid": bid,
            "ask": ask,
            "bid_volume": bid_vol,
            "ask_volume": ask_vol,
        },
        schema={
            "timestamp": pl.Datetime("ms", "UTC"),
            "bid": pl.Float64,
            "ask": pl.Float64,
            "bid_volume": pl.Float64,
            "ask_volume": pl.Float64,
        },
    )


@pytest.fixture
def tick_stream_small() -> pl.DataFrame:
    """~30 seconds of synthetic ticks at 100 ms cadence."""
    return _build_tick_frame(300)


@pytest.fixture
def tick_stream_medium() -> pl.DataFrame:
    """~5 minutes of ticks — enough for ≥ 5 time bars at 1-minute interval."""
    return _build_tick_frame(3000)


@pytest.fixture
def tick_stream_large() -> pl.DataFrame:
    """~50 minutes of ticks — used for the selector sweep and JB tests."""
    return _build_tick_frame(30_000)
