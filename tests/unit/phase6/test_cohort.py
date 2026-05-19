"""Phase 6 audit V1 — PBO cohort construction from the Alpha Registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from afml.core.registry import AlphaRegistryRepository
from afml.validation import compute_pbo
from afml.validation.cohort import (
    build_cohort_performance_matrices,
    count_cohort_trials,
)


@pytest.fixture
def in_memory_registry() -> AlphaRegistryRepository:
    repo = AlphaRegistryRepository("sqlite:///:memory:", wal_mode=False)
    repo.create_all()
    return repo


@pytest.mark.phase6
def test_count_cohort_trials_filters_by_family(
    in_memory_registry: AlphaRegistryRepository,
) -> None:
    """``count_cohort_trials`` must return the per-family trial count and the
    lab-wide total when unfiltered."""
    for i in range(5):
        in_memory_registry.record_experiment(
            agent_version="v0",
            asset="EURUSD",
            algorithmic_family="cusum",
            hyperparameter_vector={"threshold_mult": 1.0 + i},
            num_events_triggered=100,
        )
    for i in range(3):
        in_memory_registry.record_experiment(
            agent_version="v0",
            asset="EURUSD",
            algorithmic_family="bollinger",
            hyperparameter_vector={"window": 20 + i},
            num_events_triggered=100,
        )
    assert count_cohort_trials(in_memory_registry, asset="EURUSD", algorithmic_family="cusum") == 5
    assert (
        count_cohort_trials(in_memory_registry, asset="EURUSD", algorithmic_family="bollinger") == 3
    )
    # Unfiltered → lab-wide total (drives the DSR denominator).
    assert count_cohort_trials(in_memory_registry) == 8


@pytest.mark.phase6
def test_build_cohort_matrices_feed_compute_pbo() -> None:
    """The cohort matrices must have shape ``(n_valid_splits × n_strategies)``
    and be directly consumable by ``compute_pbo``."""
    rng = np.random.default_rng(0)
    n = 600
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    cohort: list[Callable[[], Any]] = [
        lambda: RandomForestClassifier(n_estimators=15, max_depth=3, random_state=0, n_jobs=1),
        lambda: RandomForestClassifier(n_estimators=15, max_depth=2, random_state=1, n_jobs=1),
        lambda: DecisionTreeClassifier(max_depth=3, random_state=2),
    ]
    is_perf, oos_perf = build_cohort_performance_matrices(
        cohort, X, y, t0, t1, n_groups=5, n_test_groups=2, embargo_pct=0.01
    )
    assert is_perf.shape[1] == 3
    assert is_perf.shape == oos_perf.shape
    assert is_perf.shape[0] >= 1
    # Must be consumable by compute_pbo.
    result = compute_pbo(is_perf, oos_perf)
    assert 0.0 <= result.pbo <= 1.0
    assert result.n_strategies == 3


@pytest.mark.phase6
def test_build_cohort_rejects_single_strategy() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 1
    with pytest.raises(ValueError, match=r"cohort|>= 2"):
        build_cohort_performance_matrices(
            [lambda: RandomForestClassifier(n_estimators=5, random_state=0)],
            X,
            y,
            t0,
            t1,
            n_groups=4,
            n_test_groups=2,
        )
