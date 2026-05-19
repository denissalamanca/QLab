"""Phase 2 — Triple-Barrier labeling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from afml.labeling.triple_barrier import apply_triple_barrier
from afml.labeling.volatility import ewm_volatility


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


# ----------------------------------------------------------------------------
# AFML audit Vulnerability 1 — Triple-Barrier intra-bar path-dependency tests
# ----------------------------------------------------------------------------
def _bars_from_ohlc(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> pl.DataFrame:
    n = close.size
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp": [t0 + timedelta(minutes=i) for i in range(n)],
            "open": close,
            "high": high,
            "low": low,
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


def _flat_path_with_spike(
    *,
    n: int = 200,
    event_idx: int = 100,
    spike_idx: int = 110,
    upper_spike_mult: float | None = None,
    lower_spike_mult: float | None = None,
    pre_event_vol: float = 0.001,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a controlled OHLC path: noisy pre-event returns drive EWM-vol,
    then a flat close trajectory after the event so no NORMAL bar hits either
    barrier. Optional spike at ``spike_idx`` plants only the high (and/or low)
    at a level guaranteed to pierce the planted barrier multiple.

    Returns (close, high, low) arrays.
    """
    rng = np.random.default_rng(seed)
    close = np.empty(n, dtype=np.float64)
    close[: event_idx + 1] = 100.0 + np.cumsum(rng.standard_normal(event_idx + 1) * pre_event_vol)
    # Post-event close is constant — only the planted spike can move things.
    close[event_idx + 1 :] = close[event_idx]

    high = close.copy()
    low = close.copy()

    log_returns = np.diff(np.log(close), prepend=np.nan)
    sigma = float(ewm_volatility(log_returns, span=20)[event_idx])

    if upper_spike_mult is not None:
        upper_barrier = close[event_idx] * (1.0 + sigma)
        high[spike_idx] = upper_barrier * upper_spike_mult
    if lower_spike_mult is not None:
        lower_barrier = close[event_idx] * (1.0 - sigma)
        low[spike_idx] = lower_barrier * lower_spike_mult

    return close, high, low


@pytest.mark.phase2
def test_intra_bar_high_triggers_upper_even_if_close_inside() -> None:
    """AFML audit V1 — intra-bar wick through the upper barrier triggers a
    take-profit hit even when the bar closes back inside the channel.

    Construction: a deterministic flat post-event close trajectory (no normal
    bar can touch a barrier) with a single ``high`` spike at index 110 that
    pierces the upper barrier. The old close-only implementation would have
    let the vertical barrier expire; the correct intra-bar implementation
    registers the touch.
    """
    close, high, low = _flat_path_with_spike(upper_spike_mult=1.5)
    bars = _bars_from_ohlc(close, high, low)

    events = pl.DataFrame(
        {"timestamp": [bars["timestamp"][100]], "side": ["long"]},
        schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8},
    )
    out = apply_triple_barrier(
        bars,
        events,
        vol_span=20,
        profit_take_mult=1.0,
        stop_loss_mult=1.0,
        vertical_barrier_bars=30,
    )
    row = out.df.row(0, named=True)
    assert row["barrier_hit"] == "upper"
    assert row["label"] == 1


@pytest.mark.phase2
def test_intra_bar_dual_touch_resolves_to_stop_loss() -> None:
    """AFML audit V1 — Conflict-Resolution Rule.

    If a single bar's high ≥ upper AND low ≤ lower (a dual-touch wide bar —
    common in flash-crash / volatility-spike conditions), label must be 0
    regardless of side. We penalize ambiguity to enforce conservative risk
    modeling.
    """
    close, high, low = _flat_path_with_spike(
        spike_idx=108,
        upper_spike_mult=2.0,
        lower_spike_mult=0.5,
    )
    bars = _bars_from_ohlc(close, high, low)

    for side_label in ("long", "short"):
        events = pl.DataFrame(
            {"timestamp": [bars["timestamp"][100]], "side": [side_label]},
            schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8},
        )
        out = apply_triple_barrier(
            bars,
            events,
            vol_span=20,
            profit_take_mult=1.0,
            stop_loss_mult=1.0,
            vertical_barrier_bars=30,
        )
        row = out.df.row(0, named=True)
        assert row["label"] == 0, f"dual-touch bar must resolve as stop-loss for side={side_label}"


@pytest.mark.phase2
def test_exit_price_is_barrier_price_not_close() -> None:
    """AFML audit V1 — exit price recorded must be the barrier level (the
    realistic fill on a stop or take-profit order), not the bar's close
    which could be far away after a price spike.
    """
    close, high, low = _flat_path_with_spike(upper_spike_mult=3.0)
    bars = _bars_from_ohlc(close, high, low)

    events = pl.DataFrame(
        {"timestamp": [bars["timestamp"][100]], "side": ["long"]},
        schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8},
    )
    out = apply_triple_barrier(
        bars,
        events,
        vol_span=20,
        profit_take_mult=1.0,
        stop_loss_mult=1.0,
        vertical_barrier_bars=30,
    )
    row = out.df.row(0, named=True)
    # Exit at the upper barrier (where the stop / TP order would fill), not at
    # high[110] nor close[110].
    assert row["exit_price"] == pytest.approx(row["upper_price"])


@pytest.mark.phase2
def test_exit_timestamp_column_present() -> None:
    """AFML 0-4 integration audit V1 — output schema must include the
    realized barrier-touch ``exit_timestamp`` so Phase 4's PurgedKFold can
    purge against the *actual* label-resolution horizon, not just the
    conservative ``vertical_timestamp``."""
    close, high, low = _flat_path_with_spike(upper_spike_mult=3.0)
    bars = _bars_from_ohlc(close, high, low)
    events = pl.DataFrame(
        {"timestamp": [bars["timestamp"][100]], "side": ["long"]},
        schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8},
    )
    out = apply_triple_barrier(
        bars,
        events,
        vol_span=20,
        vertical_barrier_bars=30,
    )
    assert "exit_timestamp" in out.df.columns
    # And the dtype must be Datetime-comparable to the bar grid (the bar tz is
    # dropped when polars rebuilds the column from indexed numpy datetime64;
    # downstream ``align_*`` helpers re-cast as needed, so we only assert the
    # time-unit precision here).
    exit_dtype = out.df.schema["exit_timestamp"]
    bar_dtype = bars.schema["timestamp"]
    assert isinstance(exit_dtype, pl.Datetime)
    assert isinstance(bar_dtype, pl.Datetime)
    assert exit_dtype.time_unit == bar_dtype.time_unit


@pytest.mark.phase2
def test_exit_timestamp_le_vertical_timestamp_always() -> None:
    """``exit_timestamp ≤ vertical_timestamp`` for every event — equality
    holds iff ``barrier_hit == 'vertical'``."""
    rng = np.random.default_rng(0)
    n = 600
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.02)
    bars = _bars_from_close(close)
    # Many overlapping events along the path so we get a mix of barrier hits.
    event_ts = [bars["timestamp"][i] for i in range(50, 500, 20)]
    events = pl.DataFrame(
        {"timestamp": event_ts, "side": ["long"] * len(event_ts)},
        schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8},
    )
    out = apply_triple_barrier(bars, events, vol_span=30, vertical_barrier_bars=20)
    rows = out.df.to_dicts()
    assert rows  # sanity — events did label
    for row in rows:
        assert row["exit_timestamp"] <= row["vertical_timestamp"], f"exit > vertical for row {row}"
        if row["barrier_hit"] == "vertical":
            assert row["exit_timestamp"] == row["vertical_timestamp"]
        else:
            assert row["exit_timestamp"] <= row["vertical_timestamp"]


@pytest.mark.phase2
def test_exit_timestamp_strictly_below_vertical_on_upper_hit() -> None:
    """When the upper barrier is touched mid-horizon, ``exit_timestamp`` must
    be strictly LESS than ``vertical_timestamp`` — proving the realized t1 is
    tighter than the worst-case t1 and can recover training data downstream."""
    close, high, low = _flat_path_with_spike(upper_spike_mult=3.0)
    bars = _bars_from_ohlc(close, high, low)
    events = pl.DataFrame(
        {"timestamp": [bars["timestamp"][100]], "side": ["long"]},
        schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8},
    )
    out = apply_triple_barrier(
        bars,
        events,
        vol_span=20,
        vertical_barrier_bars=30,
    )
    row = out.df.row(0, named=True)
    assert row["barrier_hit"] == "upper"
    assert row["exit_timestamp"] < row["vertical_timestamp"], (
        "realized t1 must be < vertical t1 when the upper barrier is hit mid-horizon"
    )
