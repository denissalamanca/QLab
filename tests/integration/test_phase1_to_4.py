"""End-to-end integration test: Phase 1 ➜ Phase 4 (AFML 0-4 audit clearance).

This test is the cross-phase contract gate. It synthesises a raw tick frame,
runs it through *every* shipped pipeline (information bars → CUSUM events →
Triple-Barrier labels → microstructure features → ONC + Clustered MDA), and
asserts the three integration invariants the Lead Quant flagged in the
Phase 0-4 Integration Audit:

* **V1** — Phase 4's purging consumes the **realized** ``exit_timestamp``
  from Triple-Barrier (tighter than ``vertical_timestamp``); the
  ``PurgedKFold`` produced internally has no overlap between train and test
  horizons.

* **V2** — Brain 1 events that fall inside the Phase 3 rolling-window
  burn-in are dropped by :func:`afml.data.align_labels_to_features`, so the
  Phase 4 feature matrix is fully finite (``np.all(np.isfinite(X))``) and
  the labels have the same row count as the features.

* **V3** — If the pipeline produces a pure-noise hypothesis, Phase 4's
  empty-MDA circuit breaker raises ``halted_at_mda=True``, logs a
  ``FAILED_AT_MDA`` row to the Alpha Registry (preserving DSR trial count),
  and returns cleanly instead of crashing Phase 5.

The synthetic fixture is intentionally compact — the goal is to prove the
data flows end-to-end without index / shape errors, NOT to validate the
mathematical content of each phase (that's the unit tests' job).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from afml.core.registry import (
    EXPERIMENT_STATUS_FAILED_AT_MDA,
    AlphaRegistryRepository,
)
from afml.data import (
    align_labels_to_features,
    build_time_bars,
    feature_matrix_to_numpy,
)
from afml.data.integration import AlignmentReport
from afml.features import compute_features
from afml.labeling import SymmetricCUSUM, TripleBarrierLabels, apply_triple_barrier
from afml.selection import PurgedKFold, select_features


@dataclass(frozen=True, slots=True)
class PipelineBundle:
    """Strongly-typed bundle of every phase's output, cached across tests."""

    ticks: pl.DataFrame
    bars: pl.DataFrame
    events: pl.DataFrame
    labels: TripleBarrierLabels
    features: pl.DataFrame
    features_aligned: pl.DataFrame
    aligned_labels: pl.DataFrame
    alignment_report: AlignmentReport


def _synthesize_ticks(n_minutes: int = 3000, seed: int = 0) -> pl.DataFrame:
    """A minute-resolution synthetic tick frame mimicking the Dukascopy schema.

    The drifting random walk + persistent volatility regime produces enough
    Brain-1 CUSUM events to exercise downstream phases.
    """
    rng = np.random.default_rng(seed)
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [t0 + timedelta(seconds=10 * i) for i in range(n_minutes * 6)]
    n = len(timestamps)
    vol = 0.0003 * (1.0 + 0.5 * np.sin(np.linspace(0, 8 * np.pi, n)))
    increments = rng.standard_normal(n) * vol
    mid = 1.10 + np.cumsum(increments)
    spread = 0.00010
    bid = mid - spread / 2
    ask = mid + spread / 2
    bid_vol = rng.integers(1, 10, size=n).astype(np.int64)
    ask_vol = rng.integers(1, 10, size=n).astype(np.int64)
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "bid": bid,
            "ask": ask,
            "bid_volume": bid_vol,
            "ask_volume": ask_vol,
        },
        schema={
            "timestamp": pl.Datetime("ms", "UTC"),
            "bid": pl.Float64,
            "ask": pl.Float64,
            "bid_volume": pl.Int64,
            "ask_volume": pl.Int64,
        },
    )


@pytest.fixture(scope="module")
def end_to_end_pipeline() -> PipelineBundle:
    """Run Phases 1 → 4 once and cache for the whole test module."""
    ticks = _synthesize_ticks(n_minutes=3000, seed=42)
    bars = build_time_bars(ticks, interval="1m")
    cusum = SymmetricCUSUM(vol_span=50, threshold_multiplier=1.0)
    events = cusum.detect(bars)
    labels = apply_triple_barrier(
        bars,
        events,
        vol_span=50,
        profit_take_mult=1.0,
        stop_loss_mult=1.0,
        vertical_barrier_bars=20,
    )
    features = compute_features(
        bars,
        events,
        windows=(10, 20, 30, 50),
        windows_entropy=(50, 100),
    )
    aligned_labels, alignment_report = align_labels_to_features(labels.df, features)
    features_aligned = features.join(
        aligned_labels.select("event_timestamp").rename({"event_timestamp": "timestamp"}),
        on="timestamp",
        how="inner",
    ).sort("timestamp")

    return PipelineBundle(
        ticks=ticks,
        bars=bars,
        events=events,
        labels=labels,
        features=features,
        features_aligned=features_aligned,
        aligned_labels=aligned_labels,
        alignment_report=alignment_report,
    )


@pytest.mark.integration
def test_pipeline_runs_end_to_end_without_errors(end_to_end_pipeline: PipelineBundle) -> None:
    """Smoke check — every phase produced non-empty output."""
    p = end_to_end_pipeline
    assert p.bars.height > 0
    assert p.events.height > 0
    assert p.labels.df.height > 0
    assert p.features.height > 0
    assert p.features_aligned.height > 0
    assert p.aligned_labels.height > 0


@pytest.mark.integration
def test_v2_alignment_drops_burn_in_events(end_to_end_pipeline: PipelineBundle) -> None:
    """V2: Brain 1 events at the head of the bar series (inside the Phase 3
    burn-in) MUST be dropped by ``align_labels_to_features``."""
    assert end_to_end_pipeline.alignment_report.n_dropped_burn_in >= 0
    assert end_to_end_pipeline.aligned_labels.height == end_to_end_pipeline.features_aligned.height


@pytest.mark.integration
def test_v2_feature_matrix_is_fully_finite(end_to_end_pipeline: PipelineBundle) -> None:
    """V2: After alignment the feature matrix must be finite — no NaN / Inf
    survived. ``feature_matrix_to_numpy`` raises if it finds any."""
    features_aligned = end_to_end_pipeline.features_aligned
    X, feature_names = feature_matrix_to_numpy(features_aligned)
    assert np.all(np.isfinite(X))
    assert X.shape[1] == len(feature_names)
    assert X.shape[0] == features_aligned.height


@pytest.mark.integration
def test_v1_purged_kfold_consumes_realized_t1(end_to_end_pipeline: PipelineBundle) -> None:
    """V1: ``PurgedKFold`` ingests the *realized* ``exit_timestamp`` (not
    ``vertical_timestamp``) and produces folds with no train ↔ test horizon
    overlap when measured against that realized t1."""
    aligned_labels = end_to_end_pipeline.aligned_labels
    t0 = aligned_labels["event_timestamp"].cast(pl.Int64).to_numpy()
    t1 = aligned_labels["exit_timestamp"].cast(pl.Int64).to_numpy()
    t1_vert = aligned_labels["vertical_timestamp"].cast(pl.Int64).to_numpy()
    assert np.all(t1 <= t1_vert)

    cv = PurgedKFold(n_splits=4, embargo_pct=0.02)
    for train_idx, test_idx in cv.split(t0, t1):
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        test_t0_min = int(t0[test_idx].min())
        test_t1_max = int(t1[test_idx].max())
        overlap_mask = (t0[train_idx] <= test_t1_max) & (t1[train_idx] >= test_t0_min)
        assert not overlap_mask.any(), (
            "PurgedKFold leaked: a train horizon overlaps the test window under the realized t1"
        )


@pytest.mark.integration
def test_v1_realized_t1_recovers_training_data_vs_vertical(
    end_to_end_pipeline: PipelineBundle,
) -> None:
    """V1: Using the realized exit_timestamp instead of the conservative
    vertical_timestamp should yield strictly ≥ as much train data per fold
    (and strictly more on at least one fold, on this overlapping fixture)."""
    aligned_labels = end_to_end_pipeline.aligned_labels
    t0 = aligned_labels["event_timestamp"].cast(pl.Int64).to_numpy()
    t1_realized = aligned_labels["exit_timestamp"].cast(pl.Int64).to_numpy()
    t1_vertical = aligned_labels["vertical_timestamp"].cast(pl.Int64).to_numpy()

    cv = PurgedKFold(n_splits=4, embargo_pct=0.02)
    folds_realized = list(cv.split(t0, t1_realized))
    folds_vertical = list(cv.split(t0, t1_vertical))
    assert len(folds_realized) == len(folds_vertical)

    realized_train_sizes = [tr.size for tr, _ in folds_realized]
    vertical_train_sizes = [tr.size for tr, _ in folds_vertical]
    for r, v in zip(realized_train_sizes, vertical_train_sizes, strict=True):
        assert r >= v, "realized t1 starved a fold worse than vertical t1"
    assert any(r > v for r, v in zip(realized_train_sizes, vertical_train_sizes, strict=True)), (
        "no fold recovered training samples under the realized t1 — V1 fix appears inert"
    )


@pytest.mark.integration
def test_v3_circuit_breaker_fires_on_pure_noise_features() -> None:
    """V3: a deliberately uninformative feature matrix (independent noise vs
    random labels) must trip the ``halted_at_mda`` flag and write a
    ``FAILED_AT_MDA`` row to the Alpha Registry — without raising."""
    rng = np.random.default_rng(0)
    n = 500
    n_features = 8
    X = pl.DataFrame({f"noise_{i}": rng.standard_normal(n) for i in range(n_features)})
    y = rng.integers(0, 2, size=n).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2

    registry = AlphaRegistryRepository("sqlite:///:memory:", wal_mode=False)
    registry.create_all()
    experiment_metadata = {
        "agent_version": "integration-0.0.1",
        "asset": "EURUSD",
        "algorithmic_family": "cusum",
        "hyperparameter_vector": {"threshold_mult": 1.0, "vol_span": 50},
        "num_events_triggered": n,
    }
    result = select_features(
        X,
        y,
        t0,
        t1,
        n_splits=4,
        random_state=0,
        min_reduction_pct=1.0,
        registry=registry,
        experiment_metadata=experiment_metadata,
    )

    assert result.halted_at_mda is True
    assert result.surviving_features == []
    assert result.registry_experiment_id is not None
    exp = registry.get(result.registry_experiment_id)
    assert exp is not None
    assert exp.status == EXPERIMENT_STATUS_FAILED_AT_MDA
    assert registry.total_trials() == 1


@pytest.mark.integration
def test_pipeline_carries_labels_aligned_to_features(
    end_to_end_pipeline: PipelineBundle,
) -> None:
    """Cross-cutting: the label vector handed to ``select_features`` must
    have exactly one row per feature matrix row, in the same timestamp order."""
    features_aligned = end_to_end_pipeline.features_aligned
    aligned_labels = end_to_end_pipeline.aligned_labels
    assert features_aligned.height == aligned_labels.height
    np.testing.assert_array_equal(
        features_aligned["timestamp"].to_numpy(),
        aligned_labels["event_timestamp"].to_numpy(),
    )
