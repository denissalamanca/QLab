"""Phase 3 test fixtures — synthetic OHLCV bars."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest


def _make_bars(
    close: np.ndarray, *, high_offset: float = 0.01, low_offset: float = 0.01
) -> pl.DataFrame:
    n = close.shape[0]
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    ts = [t0 + timedelta(minutes=i) for i in range(n)]
    rng = np.random.default_rng(0)
    high = close + np.abs(rng.normal(high_offset, high_offset * 0.1, size=n))
    low = close - np.abs(rng.normal(low_offset, low_offset * 0.1, size=n))
    volume = rng.uniform(1.0, 10.0, size=n)
    return pl.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "n_ticks": [1] * n,
        },
        schema={
            "timestamp": pl.Datetime("ms", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "n_ticks": pl.Int64,
        },
    )


@pytest.fixture
def bars_random_walk() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 3000
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
    return _make_bars(close)


@pytest.fixture
def bars_monotonic_up() -> pl.DataFrame:
    """Strict monotonic upward drift — sign sequence is all +1."""
    n = 1000
    close = 100.0 + np.arange(n) * 0.1
    return _make_bars(close)


@pytest.fixture
def bars_alternating() -> pl.DataFrame:
    """Strict up/down alternation — sign sequence flips every bar."""
    n = 1000
    close = 100.0 + np.tile([0.05, 0.0], n // 2 + 1)[:n]
    return _make_bars(close)


@pytest.fixture
def bars_long() -> pl.DataFrame:
    """Long random walk (~5000 bars) — enough warm-up for window=1000."""
    rng = np.random.default_rng(7)
    n = 5000
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
    return _make_bars(close)


@pytest.fixture
def synthetic_events(bars_long: pl.DataFrame) -> pl.DataFrame:
    """50 events sampled at evenly-spaced bar indices."""
    rng = np.random.default_rng(0)
    n = bars_long.height
    indices = np.linspace(1500, n - 100, num=50, dtype=int)
    ts = bars_long["timestamp"].to_numpy()[indices]
    sides = rng.choice(["long", "short"], size=50)
    return pl.DataFrame(
        {"timestamp": ts, "side": sides.tolist()},
        schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8},
    )
