"""End-to-end integration test: Ops M1 research sweep + certification.

Two complementary slices (the M1.8 gate):

* **Sweep flows end-to-end** — synthetic Dukascopy-shaped ticks → time bars →
  :class:`AssetPrecompute` → :func:`run_sweep` over the full CUSUM grid. On a
  near-efficient random walk the microstructure features carry no meta-label
  signal, so every config trips Phase 4's empty-MDA breaker and the plateau
  selector returns *no stable configuration* — the correct, expected outcome.
  The test asserts the orchestration is sound: 30 configs scored, each logged
  to the Alpha Registry (preserving the DSR ``K`` count), no index/shape error,
  and a graceful ``selected=None``.

* **Certification reaches the Phase-6 gate** — a random walk never survives MDA,
  so to exercise :func:`validate_strategy` we feed :func:`certify` an
  *engineered* :class:`EventDataset` whose features genuinely predict the label
  (the data-prep seam ``build_event_dataset`` is stubbed; it is independently
  covered by ``run_trial`` on real bars in the sweep slice). This drives the
  real ``select_features`` → fast-classifier CPCV cohort → PBO / DSR /
  target-shuffling → ``record_validation`` chain and asserts the gate produces a
  populated verdict and writes ``(pbo, dsr)`` back to the registry row.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import numpy as np
import polars as pl
import pytest

from afml.core.registry import AlphaRegistryRepository
from afml.data import build_time_bars
from afml.research import sweep as sweep_mod
from afml.research.harness import EventDataset
from afml.research.precompute import AssetPrecompute
from afml.research.sweep import (
    STATUS_CERTIFIED,
    STATUS_REJECTED,
    certify,
    run_sweep,
)


def _synth_ticks(n_minutes: int = 4000, seed: int = 7) -> pl.DataFrame:
    """Dukascopy-shaped synthetic ticks: AR(1) random walk with a vol regime."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    n = n_minutes * 6
    ts = [t0 + timedelta(seconds=10 * i) for i in range(n)]
    vol = 0.0004 * (1.0 + 0.6 * np.sin(np.linspace(0, 12 * np.pi, n)))
    inc = rng.standard_normal(n) * vol
    inc = inc + 0.25 * np.concatenate([[0.0], inc[:-1]])
    mid = 1.10 + np.cumsum(inc)
    spread = 0.0001
    return pl.DataFrame(
        {
            "timestamp": ts,
            "bid": mid - spread / 2,
            "ask": mid + spread / 2,
            "bid_volume": rng.integers(1, 10, n).astype(np.int64),
            "ask_volume": rng.integers(1, 10, n).astype(np.int64),
        },
        schema={
            "timestamp": pl.Datetime("ms", "UTC"),
            "bid": pl.Float64,
            "ask": pl.Float64,
            "bid_volume": pl.Int64,
            "ask_volume": pl.Int64,
        },
    )


def _precompute_from_bars(bars: pl.DataFrame, asset: str = "SYNTH") -> AssetPrecompute:
    """Wrap synthetic bars in an AssetPrecompute (FFD fields are metadata only)."""
    return AssetPrecompute(
        asset=asset,
        bars=bars,
        bar_type="time",
        bar_parameter="5m",
        bar_jarque_bera=10.0,
        n_bars=bars.height,
        regime_name="day",
        bar_hours=0.0833,
        vertical_bars=20,
        target_bar_count=bars.height,
        ffd_d=0.3,
        ffd_window=10,
        ffd_adf_pvalue=0.01,
    )


def _engineered_dataset(n: int = 600, seed: int = 0) -> EventDataset:
    """An EventDataset whose informative features genuinely predict the label."""
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(n)
    cols: dict[str, np.ndarray] = {"timestamp": np.arange(n, dtype=np.int64)}
    for i in range(3):  # strong-signal features (low noise) → survive MDA
        cols[f"info_{i}"] = signal + 0.2 * rng.standard_normal(n)
    for i in range(7):  # pure-noise features → pruned by MDA
        cols[f"noise_{i}"] = rng.standard_normal(n)
    features_aligned = pl.DataFrame(cols)
    y = (signal > 0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64) * 10
    t1 = t0 + 5
    base = np.abs(rng.standard_normal(n)) * 0.01
    return_pct = np.where(y == 1, base, -base)  # long bets earn when the label is 1
    side_sign = np.ones(n, dtype=np.int64)
    return EventDataset(
        features_aligned=features_aligned,
        y=y,
        t0=t0,
        t1=t1,
        return_pct=return_pct,
        side_sign=side_sign,
        n_alpha_events=n,
        periods_per_year=252.0,
    )


def _seed_cohort_trials(
    registry: AlphaRegistryRepository, *, asset: str, family: str, k: int = 30
) -> UUID:
    """Log ``k`` cohort trials so DSR is not cold-start quarantined (K ≥ 30)."""
    winner_id = registry.record_experiment(
        agent_version="m1-integration",
        asset=asset,
        algorithmic_family=family,
        hyperparameter_vector={"winner": True},
        num_events_triggered=600,
        brain_2_log_loss=0.23,
    )
    for i in range(k - 1):
        registry.record_experiment(
            agent_version="m1-integration",
            asset=asset,
            algorithmic_family=family,
            hyperparameter_vector={"seed": i},
            num_events_triggered=100,
            brain_2_log_loss=0.24,
        )
    return winner_id


@pytest.mark.integration
def test_m1_sweep_flows_end_to_end_on_noise() -> None:
    """Full CUSUM sweep on a random walk: orchestration sound, no stable plateau."""
    bars = build_time_bars(_synth_ticks(), interval="5m")
    pc = _precompute_from_bars(bars)
    registry = AlphaRegistryRepository("sqlite:///:memory:", wal_mode=False)
    registry.create_all()

    result = run_sweep(
        pc, "cusum", registry=registry, min_events=80, max_events=2000, n_estimators=60
    )

    assert len(result.trials) == 30  # whole grid scored
    assert len(result.surface) == 30
    # A near-efficient random walk yields no robust configuration.
    assert result.plateau.selected is None
    assert result.winner_trial is None
    # Every config that reached the model is logged (drives the DSR K count).
    logged = sum(1 for t in result.trials if t.experiment_id is not None)
    assert registry.total_trials() == logged
    assert logged >= 1


@pytest.mark.integration
def test_m1_certify_reaches_phase6_gate_on_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """certify drives select_features → CPCV cohort → PBO/DSR on engineered signal."""
    dataset = _engineered_dataset()

    def _fake_build(*_: object, **__: object) -> tuple[EventDataset, int, str]:
        return dataset, dataset.n_alpha_events, "ok"

    monkeypatch.setattr(sweep_mod, "build_event_dataset", _fake_build)

    bars = build_time_bars(_synth_ticks(n_minutes=200), interval="5m")
    pc = _precompute_from_bars(bars)
    registry = AlphaRegistryRepository("sqlite:///:memory:", wal_mode=False)
    registry.create_all()
    winner_id = _seed_cohort_trials(registry, asset=pc.asset, family="cusum")

    cert = certify(
        pc,
        "cusum",
        (2, 2),
        registry=registry,
        experiment_id=winner_id,
        cohort_n_estimators=40,
        n_shuffles=8,
    )

    # MDA kept the informative features → certify reached the Phase-6 gate.
    assert cert.status in {STATUS_CERTIFIED, STATUS_REJECTED}
    assert cert.validation is not None
    assert len(cert.surviving_features) >= 1
    assert cert.n_trials >= 30  # not cold-start quarantined
    assert 0.0 <= cert.validation.pbo.pbo <= 1.0
    assert np.isfinite(cert.validation.dsr.dsr)
    # No spurious leakage flag on genuine signal.
    assert cert.status != "data_leakage"
    # The verdict was written back to the winner's registry row.
    exp = registry.get(winner_id)
    assert exp is not None and exp.pbo is not None and exp.dsr is not None
