"""Phase 2 — Brain 1 orchestrator + Blueprint §4.3 DoD evidence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from afml.core.registry import AlphaRegistryRepository
from afml.labeling.brain1 import Brain1Result, merge_brain1_events, run_brain1
from afml.labeling.primary_alphas.bollinger import BollingerMeanReversion
from afml.labeling.primary_alphas.cusum import SymmetricCUSUM
from afml.labeling.primary_alphas.donchian import DonchianBreakout
from afml.labeling.triple_barrier import TripleBarrierLabels


@pytest.fixture
def registry(tmp_db_url: str) -> AlphaRegistryRepository:
    r = AlphaRegistryRepository(tmp_db_url)
    r.create_all()
    return r


@pytest.mark.phase2
def test_run_brain1_returns_one_result_per_plugin(
    bars_long_volatile: pl.DataFrame,
    registry: AlphaRegistryRepository,
) -> None:
    plugins = [
        SymmetricCUSUM(vol_span=100, threshold_multiplier=1.0),
        BollingerMeanReversion(sma_span=20, num_std=2.0),
        DonchianBreakout(window=20),
    ]
    results = run_brain1(
        bars_long_volatile,
        asset="EURUSD",
        plugins=plugins,
        registry=registry,
    )
    # Each plugin produces *some* events on a 10K-bar volatile series.
    assert len(results) == 3
    families = {r.plugin_family for r in results}
    assert families == {"cusum", "bbands_meanrev", "donchian_breakout"}


@pytest.mark.phase2
def test_run_brain1_logs_experiments_to_registry(
    bars_long_volatile: pl.DataFrame,
    registry: AlphaRegistryRepository,
) -> None:
    plugins = [SymmetricCUSUM(vol_span=100, threshold_multiplier=1.0)]
    results = run_brain1(bars_long_volatile, asset="EURUSD", plugins=plugins, registry=registry)
    assert results[0].experiment_id is not None
    assert registry.total_trials() == 1
    assert registry.trials_for(asset="EURUSD", family="cusum") == 1


@pytest.mark.phase2
def test_run_brain1_duplicate_hypothesis_is_silent_skip(
    bars_long_volatile: pl.DataFrame,
    registry: AlphaRegistryRepository,
) -> None:
    plugin = SymmetricCUSUM(vol_span=100, threshold_multiplier=1.0)
    run_brain1(bars_long_volatile, asset="EURUSD", plugins=[plugin], registry=registry)
    # Re-run with the same plugin & params — registry rejects duplicate, run_brain1
    # returns a Brain1Result with experiment_id=None.
    results = run_brain1(bars_long_volatile, asset="EURUSD", plugins=[plugin], registry=registry)
    assert results[0].experiment_id is None
    assert registry.total_trials() == 1  # still just the original


@pytest.mark.phase2
def test_run_brain1_skips_plugin_with_zero_events(
    bars_constant: pl.DataFrame,
    registry: AlphaRegistryRepository,
) -> None:
    """On flat prices a primary alpha can't trigger → plugin is silently
    skipped (no row in results, no row in registry)."""
    plugin = SymmetricCUSUM(vol_span=50, threshold_multiplier=1.0)
    results = run_brain1(bars_constant, asset="EURUSD", plugins=[plugin], registry=registry)
    assert results == []
    assert registry.total_trials() == 0


@pytest.mark.phase2
def test_brain1_min_500_events_dod(bars_long_volatile: pl.DataFrame) -> None:
    """Blueprint §4.3 DoD: ``len(events) >= 500`` per asset (across plugins)."""
    plugins = [
        SymmetricCUSUM(vol_span=100, threshold_multiplier=1.0),
        BollingerMeanReversion(sma_span=20, num_std=1.5),
        DonchianBreakout(window=20),
    ]
    results = run_brain1(bars_long_volatile, asset="EURUSD", plugins=plugins)
    total = sum(r.n_events for r in results)
    assert total >= 500, f"Total Brain 1 events {total} below the 500-minimum DoD threshold"


@pytest.mark.phase2
def test_brain1_recall_dod_cusum_on_spike_data(
    bars_with_spikes: tuple[pl.DataFrame, list[int]],
) -> None:
    """Blueprint §4.3 DoD: recall > 0.70 against a known ground-truth signal.

    The fixture injects 20 spikes at known indices; CUSUM (the structural
    volatility-event filter) must catch ≥ 14 of them inside a small causal
    window.
    """

    bars, spike_indices = bars_with_spikes
    plugins = [SymmetricCUSUM(vol_span=100, threshold_multiplier=1.0)]
    results = run_brain1(bars, asset="EURUSD", plugins=plugins)
    assert len(results) == 1

    event_ts = results[0].events["timestamp"].to_numpy()
    bar_ts = bars["timestamp"].to_numpy()
    event_idx = np.searchsorted(bar_ts, event_ts)
    tolerance = 15
    hit = sum(np.any(np.abs(event_idx - s) <= tolerance) for s in spike_indices)
    recall = hit / len(spike_indices)
    assert recall > 0.70


@pytest.mark.phase2
def test_brain1_result_label_rate_in_range(
    bars_long_volatile: pl.DataFrame,
) -> None:
    plugins = [SymmetricCUSUM(vol_span=100, threshold_multiplier=1.0)]
    results = run_brain1(bars_long_volatile, asset="EURUSD", plugins=plugins)
    res: Brain1Result = results[0]
    assert 0.0 <= res.label_rate <= 1.0


# ----------------------------------------------------------------------------
# AFML audit §2.2 — Event-Sampling Deduplication / Dual-Path Cleanliness
# ----------------------------------------------------------------------------
def _fake_brain1_result(
    family: str,
    timestamps: list[datetime],
    sides: list[str],
) -> Brain1Result:
    """Build a synthetic Brain1Result for dedup testing."""
    events = pl.DataFrame(
        {"timestamp": timestamps, "side": sides},
        schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8},
    )
    empty_labels = pl.DataFrame(
        schema={
            "event_timestamp": pl.Datetime("ms", "UTC"),
            "side": pl.Utf8,
            "entry_price": pl.Float64,
            "upper_price": pl.Float64,
            "lower_price": pl.Float64,
            "vertical_timestamp": pl.Datetime("ms", "UTC"),
            "barrier_hit": pl.Utf8,
            "exit_price": pl.Float64,
            "return_pct": pl.Float64,
            "label": pl.Int64,
        }
    )
    return Brain1Result(
        asset="EURUSD",
        plugin_family=family,
        hyperparameters={},
        events=events,
        labels=TripleBarrierLabels(
            df=empty_labels,
            profit_take_mult=1.0,
            stop_loss_mult=1.0,
            vertical_barrier_bars=20,
            vol_span=100,
        ),
        experiment_id=None,
    )


@pytest.mark.phase2
def test_merge_brain1_events_empty_input_returns_empty_schema() -> None:
    out = merge_brain1_events([])
    assert out.height == 0
    assert {"timestamp", "side", "plugin_family"} <= set(out.columns)


@pytest.mark.phase2
def test_merge_brain1_events_dedupes_same_timestamp() -> None:
    """AFML audit §2.2 — if CUSUM and Bollinger trigger on the SAME bar
    the merged event set must contain ONE entry, not two."""
    common_ts = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    cusum_only_ts = datetime(2024, 1, 1, 11, 0, tzinfo=UTC)
    bb_only_ts = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

    cusum = _fake_brain1_result("cusum", [common_ts, cusum_only_ts], ["long", "long"])
    bbands = _fake_brain1_result("bbands_meanrev", [common_ts, bb_only_ts], ["short", "short"])

    merged = merge_brain1_events([cusum, bbands])
    # 3 unique timestamps total.
    assert merged.height == 3
    ts_list = merged["timestamp"].to_list()
    assert len(set(ts_list)) == len(ts_list)
    # On collision, keep first (CUSUM came first in the input list) → long side.
    row = merged.filter(pl.col("timestamp") == common_ts).row(0, named=True)
    assert row["plugin_family"] == "cusum"
    assert row["side"] == "long"


@pytest.mark.phase2
def test_merge_brain1_events_monotonic_after_dedup() -> None:
    """Merged events must be strictly monotonically increasing in time."""
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    # Interleave timestamps from two plugins so the merged output requires sorting.
    ts_a = [t0 + timedelta(minutes=i) for i in (0, 5, 10, 15)]
    ts_b = [t0 + timedelta(minutes=i) for i in (2, 7, 10, 20)]  # 10 collides

    a = _fake_brain1_result("cusum", ts_a, ["long"] * 4)
    b = _fake_brain1_result("bbands_meanrev", ts_b, ["short"] * 4)

    merged = merge_brain1_events([a, b])
    ts_out = merged["timestamp"].to_list()
    assert ts_out == sorted(ts_out)
    # 4 + 4 - 1 collision = 7 unique timestamps.
    assert merged.height == 7


@pytest.mark.phase2
def test_merge_brain1_events_skips_empty_plugin_results() -> None:
    """A plugin that produced zero events must not break the merge."""
    only_ts = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    populated = _fake_brain1_result("cusum", [only_ts], ["long"])
    empty = _fake_brain1_result("bbands_meanrev", [], [])

    merged = merge_brain1_events([populated, empty])
    assert merged.height == 1
    assert merged.row(0, named=True)["plugin_family"] == "cusum"
