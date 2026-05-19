"""Phase 5 — sequential bootstrap (AFML Snippet 4.3).

The headline property AFML proves for the sequential bootstrap is that the
expected average uniqueness of the drawn sample is higher than that of a
uniform-with-replacement bootstrap. The tests below lock this in.
"""

from __future__ import annotations

import numpy as np
import pytest

from afml.modeling.concurrency import (
    average_uniqueness,
    concurrency_count,
    indicator_matrix,
)
from afml.modeling.sequential_bootstrap import sequential_bootstrap


def _bootstrap_avg_uniqueness(ind: np.ndarray, draw: np.ndarray) -> float:
    """Average uniqueness ``ū`` of the bootstrapped sample as a single scalar.

    Compute concurrency restricted to the drawn events (a sample can pick the
    same event multiple times, raising its self-concurrency) and average the
    per-event uniqueness.
    """
    boot_ind = ind[:, draw]  # (n_grid, n_draw)
    boot_c = concurrency_count(boot_ind)
    avg_u = average_uniqueness(boot_ind, boot_c)
    return float(np.mean(avg_u))


@pytest.mark.phase5
def test_sequential_bootstrap_returns_correct_shape() -> None:
    t0 = np.arange(20, dtype=np.int64)
    t1 = t0 + 3
    ind = indicator_matrix(t0, t1)
    rng = np.random.default_rng(0)
    draw = sequential_bootstrap(ind, n_samples=15, rng=rng)
    assert draw.shape == (15,)
    assert draw.dtype == np.int64
    assert draw.min() >= 0
    assert draw.max() < 20


@pytest.mark.phase5
def test_sequential_bootstrap_default_n_samples_matches_n_events() -> None:
    t0 = np.arange(10, dtype=np.int64)
    t1 = t0 + 2
    ind = indicator_matrix(t0, t1)
    draw = sequential_bootstrap(ind, rng=np.random.default_rng(0))
    assert draw.size == 10


@pytest.mark.phase5
def test_sequential_bootstrap_reproducible_with_seed() -> None:
    t0 = np.arange(30, dtype=np.int64)
    t1 = t0 + 3
    ind = indicator_matrix(t0, t1)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    d1 = sequential_bootstrap(ind, n_samples=20, rng=rng1)
    d2 = sequential_bootstrap(ind, n_samples=20, rng=rng2)
    np.testing.assert_array_equal(d1, d2)


@pytest.mark.phase5
def test_sequential_bootstrap_beats_uniform_in_average_uniqueness() -> None:
    """The central AFML claim: sequential bootstrap produces samples with
    higher average uniqueness than the naive uniform bootstrap on the same
    indicator matrix.

    We use a many-trial Monte Carlo average so the comparison is stable
    against per-draw variance.
    """
    n = 40
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 5  # heavy overlap → uniform bootstrap will repeat overlapping events
    ind = indicator_matrix(t0, t1)

    n_trials = 30
    seq_means = []
    unif_means = []
    for trial in range(n_trials):
        trial_rng = np.random.default_rng(trial)
        seq_draw = sequential_bootstrap(ind, n_samples=n, rng=trial_rng)
        unif_draw = trial_rng.integers(0, n, size=n).astype(np.int64)
        seq_means.append(_bootstrap_avg_uniqueness(ind, seq_draw))
        unif_means.append(_bootstrap_avg_uniqueness(ind, unif_draw))
    assert np.mean(seq_means) > np.mean(unif_means), (
        f"sequential bootstrap not better: seq={np.mean(seq_means):.3f}, "
        f"unif={np.mean(unif_means):.3f}"
    )


@pytest.mark.phase5
def test_sequential_bootstrap_zero_samples_returns_empty() -> None:
    t0 = np.arange(5, dtype=np.int64)
    t1 = t0 + 1
    ind = indicator_matrix(t0, t1)
    draw = sequential_bootstrap(ind, n_samples=0, rng=np.random.default_rng(0))
    assert draw.size == 0


@pytest.mark.phase5
def test_sequential_bootstrap_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        sequential_bootstrap(np.zeros(5, dtype=np.int64), n_samples=3)
    with pytest.raises(ValueError, match="≥ 0"):
        sequential_bootstrap(np.zeros((3, 4), dtype=np.int64), n_samples=-1)


@pytest.mark.phase5
def test_sequential_bootstrap_prefers_non_overlapping_events_after_first_pick() -> None:
    """Over a multi-sample draw, the isolated event should accumulate a higher
    aggregate selection count than the average cluster event.

    AFML detail: the **first** draw of a sequential bootstrap with empty
    ``phi`` is unconditionally uniform — the conditional uniqueness collapses
    to 1 for every candidate because the augmented indicator matrix has only
    one event in it. The sequential preference only materialises from the
    second pick onward, when already-drawn events suppress the cluster bars
    via the running concurrency. Hence we draw 20 samples per trial and
    average the per-event counts.
    """
    # 5 overlapping events in a tight cluster + 1 isolated event.
    t0 = np.array([0, 1, 2, 3, 4, 100], dtype=np.int64)
    t1 = np.array([10, 11, 12, 13, 14, 110], dtype=np.int64)
    ind = indicator_matrix(t0, t1)

    n_trials = 100
    samples_per_trial = 20
    counts = np.zeros(6, dtype=np.int64)
    for trial in range(n_trials):
        draw = sequential_bootstrap(
            ind, n_samples=samples_per_trial, rng=np.random.default_rng(trial)
        )
        for ev in draw:
            counts[ev] += 1
    # Per-event mean selection rate.
    total = n_trials * samples_per_trial
    isolated_rate = counts[5] / total
    cluster_rate = counts[:5].mean() / total
    assert isolated_rate > cluster_rate, (
        f"isolated event not preferred over many draws: {isolated_rate=:.3f}, {cluster_rate=:.3f}"
    )
