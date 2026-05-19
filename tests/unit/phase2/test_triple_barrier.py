"""Phase 2 — Triple-Barrier labeling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from afml.labeling.triple_barrier import apply_triple_barrier


def _bars_from_close(close: np.ndarray) -> pl.DataFrame:
    n = close.size
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp": [t0 + timedelta(minutes=i) for i in range(n)],
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        },
        schema={
            "timestamp": pl.Datetime("ms", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
    )


@pytest.mark.phase2
def test_long_upper_barrier_hit_labels_one() -> None:
    """Price drifts up after event → upper hit → long ⇒ label = 1."""
    n = 200
    rng = np.random.default_rng(0)
    # Small pre-event noise so the EWM volatility estimate is non-zero,
    # then a strong upward drift after the event.
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
    close[105:] += np.linspace(0.0, 20.0, n - 105)
    bars = _bars_from_close(close)
    events = pl.DataFrame(
        {
            "timestamp": [bars["timestamp"][100]],
            "side": ["long"],
        },
        schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8},
    )
    out = apply_triple_barrier(
        bars,
        events,
        vol_span=20,
        profit_take_mult=1.0,
        stop_loss_mult=1.0,
        vertical_barrier_bars=50,
    )
    assert out.n_events == 1
    row = out.df.row(0, named=True)
    assert row["barrier_hit"] == "upper"
    assert row["label"] == 1


@pytest.mark.phase2
def test_short_lower_barrier_hit_labels_one() -> None:
    n = 200
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
    close[105:] -= np.linspace(0.0, 20.0, n - 105)
    bars = _bars_from_close(close)
    events = pl.DataFrame(
        {
            "timestamp": [bars["timestamp"][100]],
            "side": ["short"],
        },
        schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8},
    )
    out = apply_triple_barrier(
        bars,
        events,
        vol_span=20,
        profit_take_mult=1.0,
        stop_loss_mult=1.0,
        vertical_barrier_bars=50,
    )
    row = out.df.row(0, named=True)
    assert row["barrier_hit"] == "lower"
    assert row["label"] == 1


@pytest.mark.phase2
def test_long_stopped_against_labels_zero() -> None:
    """Long position; price drifts DOWN → lower hit → label = 0."""
    n = 200
    rng = np.random.default_rng(0)
    # Some volatility for the EWM to estimate, then a strong downward drift.
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.1)
    close[105:] -= np.linspace(0.0, 30.0, n - 105)
    bars = _bars_from_close(close)
    events = pl.DataFrame({
        "timestamp": [bars["timestamp"][100]],
        "side": ["long"],
    })
    out = apply_triple_barrier(
        bars,
        events,
        vol_span=20,
        profit_take_mult=1.0,
        stop_loss_mult=1.0,
        vertical_barrier_bars=60,
    )
    row = out.df.row(0, named=True)
    assert row["barrier_hit"] == "lower"
    assert row["label"] == 0


@pytest.mark.phase2
def test_vertical_barrier_labels_zero() -> None:
    """Sideways price; no barrier hits → vertical wins → label = 0."""
    n = 200
    rng = np.random.default_rng(0)
    close = 100.0 + rng.standard_normal(n) * 0.005  # very small noise
    bars = _bars_from_close(close)
    events = pl.DataFrame({
        "timestamp": [bars["timestamp"][100]],
        "side": ["long"],
    })
    out = apply_triple_barrier(
        bars,
        events,
        vol_span=20,
        profit_take_mult=5.0,
        stop_loss_mult=5.0,
        vertical_barrier_bars=20,
    )
    row = out.df.row(0, named=True)
    assert row["barrier_hit"] == "vertical"
    assert row["label"] == 0


@pytest.mark.phase2
def test_event_outside_bar_grid_skipped() -> None:
    """Events whose timestamp doesn't align with a bar are silently dropped."""
    n = 100
    close = np.full(n, 100.0)
    bars = _bars_from_close(close)
    far_future = bars["timestamp"][n - 1] + timedelta(days=999)
    events = pl.DataFrame({
        "timestamp": [far_future],
        "side": ["long"],
    })
    out = apply_triple_barrier(bars, events, vol_span=20)
    assert out.n_events == 0


@pytest.mark.phase2
def test_vertical_barrier_bars_must_be_positive() -> None:
    bars = _bars_from_close(np.full(50, 100.0))
    events = pl.DataFrame({"timestamp": [bars["timestamp"][10]], "side": ["long"]})
    with pytest.raises(ValueError, match="vertical_barrier_bars"):
        apply_triple_barrier(bars, events, vertical_barrier_bars=0)


@pytest.mark.phase2
def test_labels_summary_counts_consistent() -> None:
    """``n_events == n_positive + n_negative`` always."""
    rng = np.random.default_rng(0)
    n = 1000
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.1)
    bars = _bars_from_close(close)
    event_indices = [200, 400, 600, 800]
    events = pl.DataFrame({
        "timestamp": [bars["timestamp"][i] for i in event_indices],
        "side": ["long", "short", "long", "short"],
    })
    out = apply_triple_barrier(bars, events, vol_span=50)
    assert out.n_events == out.n_positive + out.n_negative
