"""Phase 2 — Brain 1 orchestrator + Blueprint §4.3 DoD evidence."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from afml.core.registry import AlphaRegistryRepository
from afml.labeling.brain1 import Brain1Result, run_brain1
from afml.labeling.primary_alphas.bollinger import BollingerMeanReversion
from afml.labeling.primary_alphas.cusum import SymmetricCUSUM
from afml.labeling.primary_alphas.donchian import DonchianBreakout


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
