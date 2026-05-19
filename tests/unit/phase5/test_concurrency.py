"""Phase 5 — indicator matrix / concurrency / average uniqueness."""

from __future__ import annotations

import numpy as np
import pytest

from afml.modeling.concurrency import (
    average_uniqueness,
    concurrency_count,
    indicator_matrix,
)


@pytest.mark.phase5
def test_indicator_matrix_binary() -> None:
    t0 = np.array([0, 10, 20], dtype=np.int64)
    t1 = np.array([5, 15, 25], dtype=np.int64)
    ind = indicator_matrix(t0, t1)
    assert set(np.unique(ind).tolist()) <= {0, 1}


@pytest.mark.phase5
def test_indicator_matrix_active_window() -> None:
    """Event ``i`` is active iff ``grid_t ∈ [t0_i, t1_i]``."""
    t0 = np.array([0, 10], dtype=np.int64)
    t1 = np.array([5, 15], dtype=np.int64)
    grid = np.arange(0, 20, dtype=np.int64)
    ind = indicator_matrix(t0, t1, grid)
    for grid_i, ts in enumerate(grid):
        for ev_i in range(2):
            expected = 1 if t0[ev_i] <= ts <= t1[ev_i] else 0
            assert ind[grid_i, ev_i] == expected, f"mismatch at grid {ts}, event {ev_i}"


@pytest.mark.phase5
def test_concurrency_simple() -> None:
    """Three identical horizons ⇒ c_t = 3 on every active bar."""
    t0 = np.array([0, 0, 0], dtype=np.int64)
    t1 = np.array([4, 4, 4], dtype=np.int64)
    ind = indicator_matrix(t0, t1)
    c = concurrency_count(ind)
    assert np.all(c == 3)


@pytest.mark.phase5
def test_uniqueness_non_overlapping_events_is_one() -> None:
    """Non-overlapping horizons ⇒ ū_i = 1 for every event."""
    t0 = np.array([0, 10, 20], dtype=np.int64)
    t1 = np.array([5, 15, 25], dtype=np.int64)
    ind = indicator_matrix(t0, t1)
    avg_u = average_uniqueness(ind)
    np.testing.assert_allclose(avg_u, [1.0, 1.0, 1.0])


@pytest.mark.phase5
def test_uniqueness_identical_horizons_is_one_over_n() -> None:
    """N events with identical horizons share concurrency N at every bar ⇒
    every event has ū_i = 1/N."""
    for n in (2, 3, 4, 5):
        t0 = np.zeros(n, dtype=np.int64)
        t1 = np.full(n, 5, dtype=np.int64)
        ind = indicator_matrix(t0, t1)
        avg_u = average_uniqueness(ind)
        np.testing.assert_allclose(avg_u, np.full(n, 1.0 / n))


@pytest.mark.phase5
def test_uniqueness_in_zero_one() -> None:
    """ū_i ∈ [0, 1] under any input."""
    rng = np.random.default_rng(0)
    n = 50
    t0 = rng.integers(0, 100, size=n).astype(np.int64)
    t1 = t0 + rng.integers(1, 10, size=n).astype(np.int64)
    ind = indicator_matrix(t0, t1)
    avg_u = average_uniqueness(ind)
    assert avg_u.min() >= 0.0
    assert avg_u.max() <= 1.0


@pytest.mark.phase5
def test_indicator_matrix_rejects_inverted_horizon() -> None:
    t0 = np.array([0, 10], dtype=np.int64)
    t1 = np.array([5, 8], dtype=np.int64)  # t1[1] < t0[1]
    with pytest.raises(ValueError, match="t1"):
        indicator_matrix(t0, t1)


@pytest.mark.phase5
def test_uniqueness_partial_overlap() -> None:
    """Manual verification on the docstring example."""
    # Event 0: [0, 3], Event 1: [1, 4], Event 2: [5, 8], Event 3: [10, 13]
    # Grid (unique union) = [0, 1, 3, 4, 5, 8, 10, 13]
    # Concurrency:                1, 2, 2, 1, 1, 1,  1,  1
    # Event 0 active at grid 0, 1, 3 with c = 1, 2, 2 → u_0 = (1 + 0.5 + 0.5)/3 = 2/3
    # Event 1 active at grid 1, 3, 4 with c = 2, 2, 1 → u_1 = (0.5 + 0.5 + 1)/3 = 2/3
    # Event 2 active at grid 5, 8 with c = 1, 1     → u_2 = 1
    # Event 3 active at grid 10, 13 with c = 1, 1   → u_3 = 1
    t0 = np.array([0, 1, 5, 10], dtype=np.int64)
    t1 = np.array([3, 4, 8, 13], dtype=np.int64)
    ind = indicator_matrix(t0, t1)
    avg_u = average_uniqueness(ind)
    np.testing.assert_allclose(avg_u, [2 / 3, 2 / 3, 1.0, 1.0])
