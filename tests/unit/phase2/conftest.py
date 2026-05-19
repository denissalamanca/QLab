"""Phase 2 test fixtures — synthetic OHLC bar streams."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest


def _make_bars(
    prices: np.ndarray,
    *,
    start: datetime | None = None,
    interval_minutes: int = 1,
    vol_scale: float = 0.001,
) -> pl.DataFrame:
    """Pack a 1-D close price series into a Phase 1-shaped bars DataFrame."""
    n = prices.shape[0]
    if start is None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=i * interval_minutes) for i in range(n)]
    # Synthetic OHL = close ± small noise so bars are non-degenerate.
    rng = np.random.default_rng(0)
    noise = rng.normal(0, vol_scale, size=(n, 2))
    high = prices + np.abs(noise[:, 0])
    low = prices - np.abs(noise[:, 1])
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": high,
            "low": low,
            "close": prices,
            "volume": [1.0] * n,
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
def bars_constant() -> pl.DataFrame:
    """500 bars at a flat $100. No primary-alpha should fire."""
    return _make_bars(np.full(500, 100.0), vol_scale=0.0)


@pytest.fixture
def bars_random_walk() -> pl.DataFrame:
    """Long noisy random walk — ~5000 bars."""
    rng = np.random.default_rng(42)
    increments = rng.standard_normal(5000) * 0.01
    prices = 100.0 + np.cumsum(increments)
    return _make_bars(prices)


@pytest.fixture
def bars_with_spikes() -> tuple[pl.DataFrame, list[int]]:
    """Random walk with N=20 large injected spikes at known indices.

    The fixture also returns the spike indices so tests can compute the recall
    of a primary-alpha detector against the ground truth.
    """
    rng = np.random.default_rng(2024)
    n = 1500
    increments = rng.standard_normal(n) * 0.005
    prices = 100.0 + np.cumsum(increments)
    spike_indices = [
        100,
        175,
        250,
        325,
        400,
        475,
        550,
        625,
        700,
        775,
        850,
        925,
        1000,
        1075,
        1150,
        1225,
        1300,
        1375,
        1430,
        1480,
    ]
    sign = 1.0
    for idx in spike_indices:
        # Inject a 6-bar streak of strong moves; flip sign each spike for symmetry.
        for j in range(6):
            if idx + j < n:
                prices[idx + j :] += sign * 0.8
        sign *= -1.0
    return _make_bars(prices), spike_indices


@pytest.fixture
def bars_mean_reverting() -> pl.DataFrame:
    """OU-like mean-reverting series — useful for Bollinger plugin."""
    rng = np.random.default_rng(7)
    n = 2000
    theta = 0.05  # reversion rate
    mu = 100.0
    sigma = 0.5
    prices = np.empty(n)
    prices[0] = mu
    for t in range(1, n):
        prices[t] = prices[t - 1] + theta * (mu - prices[t - 1]) + sigma * rng.standard_normal()
    return _make_bars(prices)


@pytest.fixture
def bars_trending() -> pl.DataFrame:
    """Strong upward drift with occasional pullbacks — useful for Donchian breakout."""
    rng = np.random.default_rng(11)
    n = 2000
    increments = rng.standard_normal(n) * 0.05 + 0.05
    prices = 100.0 + np.cumsum(increments)
    return _make_bars(prices)


@pytest.fixture
def bars_long_volatile() -> pl.DataFrame:
    """Long volatile series — ~10000 bars for the ≥500-event Brain 1 DoD test."""
    rng = np.random.default_rng(99)
    n = 10_000
    increments = rng.standard_normal(n) * 0.02
    prices = 100.0 + np.cumsum(increments)
    return _make_bars(prices)
