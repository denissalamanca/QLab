"""Phase 2 — Symmetric CUSUM plugin."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from afml.labeling.primary_alphas.cusum import SymmetricCUSUM


@pytest.mark.phase2
def test_cusum_empty_on_constant_prices(bars_constant: pl.DataFrame) -> None:
    """A flat price line has zero returns → no events."""
    plugin = SymmetricCUSUM(vol_span=50, threshold_multiplier=1.0)
    events = plugin.detect(bars_constant)
    assert events.height == 0


@pytest.mark.phase2
def test_cusum_returns_schema(bars_random_walk: pl.DataFrame) -> None:
    plugin = SymmetricCUSUM(vol_span=100, threshold_multiplier=1.0)
    events = plugin.detect(bars_random_walk)
    assert {"timestamp", "side"} <= set(events.columns)
    if events.height > 0:
        assert set(events["side"].unique().to_list()) <= {"long", "short"}


@pytest.mark.phase2
def test_cusum_more_events_with_lower_threshold(bars_random_walk: pl.DataFrame) -> None:
    low = SymmetricCUSUM(vol_span=100, threshold_multiplier=0.5).detect(bars_random_walk)
    high = SymmetricCUSUM(vol_span=100, threshold_multiplier=3.0).detect(bars_random_walk)
    assert low.height >= high.height


@pytest.mark.phase2
def test_cusum_recall_on_injected_spikes(
    bars_with_spikes: tuple[pl.DataFrame, list[int]],
) -> None:
    """Blueprint §4.3 DoD: recall > 0.70 on a known-spikes ground truth.

    Injects 20 large directional moves into a noisy random walk; CUSUM with
    default vol-scaled threshold must flag at least 70 % of them within a small
    causal window after each spike's onset.
    """
    bars, spike_indices = bars_with_spikes
    plugin = SymmetricCUSUM(vol_span=100, threshold_multiplier=1.0)
    events = plugin.detect(bars)
    event_idx = np.searchsorted(
        bars["timestamp"].to_numpy(),
        events["timestamp"].to_numpy(),
    )
    tolerance = 15
    hit = 0
    for spike in spike_indices:
        if np.any(np.abs(event_idx - spike) <= tolerance):
            hit += 1
    recall = hit / len(spike_indices)
    assert recall > 0.70, f"CUSUM recall {recall:.2%} below the 70% DoD threshold"


@pytest.mark.phase2
def test_cusum_chronological_events(bars_random_walk: pl.DataFrame) -> None:
    plugin = SymmetricCUSUM(vol_span=100, threshold_multiplier=1.0)
    events = plugin.detect(bars_random_walk)
    if events.height > 1:
        ts = events["timestamp"].to_list()
        assert ts == sorted(ts)


@pytest.mark.phase2
def test_cusum_params_round_trip() -> None:
    plugin = SymmetricCUSUM(vol_span=120, threshold_multiplier=1.5)
    assert plugin.params == {"vol_span": 120, "threshold_multiplier": 1.5}
    assert plugin.algorithmic_family == "cusum"
