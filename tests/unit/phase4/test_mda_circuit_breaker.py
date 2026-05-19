"""Phase 4 — empty-MDA circuit breaker (AFML 0-4 integration audit V3).

If Brain 1 produces a pure-noise hypothesis, Clustered MDA correctly returns
zero surviving clusters. Without the circuit breaker, the orchestrator would
then either crash Phase 5 (empty feature matrix) or silently emit a useless
``SelectionResult`` that downstream callers can't distinguish from success.

The breaker forces:

1. ``SelectionResult.halted_at_mda is True``.
2. ``SelectionResult.surviving_features == []``.
3. If a registry is wired in, an immutable ``FAILED_AT_MDA`` row is recorded —
   preserving the DSR trial denominator.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from afml.core.registry import (
    EXPERIMENT_STATUS_COMPLETED,
    EXPERIMENT_STATUS_FAILED_AT_MDA,
    AlphaRegistryRepository,
)
from afml.selection.pipeline import select_features


@pytest.fixture
def in_memory_registry() -> AlphaRegistryRepository:
    repo = AlphaRegistryRepository("sqlite:///:memory:", wal_mode=False)
    repo.create_all()
    return repo


def _pure_noise_dataset() -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Tiny pure-noise frame that defeats both MDA and SFI.

    Both ``y`` and ``X`` are independent rng draws — no feature can possibly
    explain ``y``, so MDA's permutation t-test will reject every cluster.
    """
    rng = np.random.default_rng(0)
    n = 400
    n_features = 8
    cols = {f"noise_{i}": rng.standard_normal(n) for i in range(n_features)}
    X = pl.DataFrame(cols)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    return X, y, t0, t1


@pytest.mark.phase4
def test_pure_noise_triggers_circuit_breaker_flag() -> None:
    """Pure noise + correctly-rejected clusters + min_reduction_pct=1.0
    (impossible to satisfy) ⇒ MDA empty, SFI hard-capped to 0 ⇒ halted."""
    X, y, t0, t1 = _pure_noise_dataset()
    result = select_features(X, y, t0, t1, n_splits=4, random_state=0, min_reduction_pct=1.0)
    assert result.halted_at_mda is True
    assert result.surviving_features == []


@pytest.mark.phase4
def test_circuit_breaker_logs_failed_at_mda_to_registry(
    in_memory_registry: AlphaRegistryRepository,
) -> None:
    """When a registry is supplied alongside experiment metadata, the
    empty-survivor case must log a ``FAILED_AT_MDA`` row and return the new
    experiment UUID via ``SelectionResult.registry_experiment_id``."""
    X, y, t0, t1 = _pure_noise_dataset()
    metadata = {
        "agent_version": "audit-remediation-0.0.1",
        "asset": "EURUSD",
        "algorithmic_family": "cusum",
        "hyperparameter_vector": {"threshold_mult": 2.0, "vol_span": 100},
        "num_events_triggered": int(y.size),
    }
    result = select_features(
        X,
        y,
        t0,
        t1,
        n_splits=4,
        random_state=0,
        min_reduction_pct=1.0,
        registry=in_memory_registry,
        experiment_metadata=metadata,
    )
    assert result.halted_at_mda is True
    assert result.registry_experiment_id is not None
    # Pull the record back via the repository and check the status.
    exp = in_memory_registry.get(result.registry_experiment_id)
    assert exp is not None
    assert exp.status == EXPERIMENT_STATUS_FAILED_AT_MDA
    assert exp.brain_2_log_loss is None
    assert exp.is_deployed is False
    # DSR trial-count integrity — failed trial still counted.
    assert in_memory_registry.total_trials() == 1


@pytest.mark.phase4
def test_circuit_breaker_skips_registry_when_not_provided() -> None:
    """Without a registry, the halt is silent (no exceptions)."""
    X, y, t0, t1 = _pure_noise_dataset()
    result = select_features(X, y, t0, t1, n_splits=4, random_state=0, min_reduction_pct=1.0)
    assert result.halted_at_mda is True
    assert result.registry_experiment_id is None


@pytest.mark.phase4
def test_circuit_breaker_does_not_fire_when_survivors_exist(
    in_memory_registry: AlphaRegistryRepository,
) -> None:
    """A signal-bearing dataset must NOT trigger the breaker, and must NOT
    write a FAILED_AT_MDA row to the registry."""
    rng = np.random.default_rng(20260520)
    n = 800
    n_features = 10
    latent = rng.standard_normal(n)
    y = (latent > 0.0).astype(np.int64)
    cols = {
        f"redundant_{i}": latent + rng.standard_normal(n) * (0.25 + 0.05 * i)
        for i in range(n_features)
    }
    X = pl.DataFrame(cols)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    metadata = {
        "agent_version": "audit-remediation-0.0.1",
        "asset": "EURUSD",
        "algorithmic_family": "cusum",
        "hyperparameter_vector": {"threshold_mult": 3.0, "vol_span": 50},
        "num_events_triggered": int(y.size),
    }
    result = select_features(
        X,
        y,
        t0,
        t1,
        n_splits=4,
        random_state=0,
        registry=in_memory_registry,
        experiment_metadata=metadata,
    )
    assert result.halted_at_mda is False
    assert result.registry_experiment_id is None
    # No row should have been written.
    assert in_memory_registry.total_trials() == 0


@pytest.mark.phase4
def test_registry_default_status_is_completed(
    in_memory_registry: AlphaRegistryRepository,
) -> None:
    """Sanity: ``record_experiment`` without an explicit status defaults to
    ``EXPERIMENT_STATUS_COMPLETED`` — backward compatibility with Phase 0-2
    tests that don't pass a status."""
    eid = in_memory_registry.record_experiment(
        agent_version="v0",
        asset="EURUSD",
        algorithmic_family="cusum",
        hyperparameter_vector={"x": 1},
        num_events_triggered=10,
    )
    exp = in_memory_registry.get(eid)
    assert exp is not None
    assert exp.status == EXPERIMENT_STATUS_COMPLETED


@pytest.mark.phase4
def test_registry_rejects_unknown_status(in_memory_registry: AlphaRegistryRepository) -> None:
    with pytest.raises(ValueError, match="unknown status"):
        in_memory_registry.record_experiment(
            agent_version="v0",
            asset="EURUSD",
            algorithmic_family="cusum",
            hyperparameter_vector={"x": 1},
            num_events_triggered=10,
            status="UNKNOWN",
        )
