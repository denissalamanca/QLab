"""M1.6 — sweep orchestration + two-stage certification.

These are *orchestration* unit tests: the heavy per-config pipeline (``run_trial``)
and the Phase 6 gate (``validate_strategy``) are exercised end-to-end on real
ticks by the M1.8 integration test. Here we stub those seams and assert the
sweep wiring — surface assembly, plateau selection, winner look-up, certification
dispatch, and the typed early-exit statuses — is correct and deterministic.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl
import pytest

from afml.core.registry import AlphaRegistryRepository
from afml.research import sweep as sweep_mod
from afml.research.grids import get_family_grid
from afml.research.harness import TrialResult
from afml.research.plateau import Coord
from afml.research.precompute import AssetPrecompute
from afml.research.sweep import (
    STATUS_CERTIFIED,
    STATUS_INSUFFICIENT_EVENTS,
    STATUS_REJECTED,
    CertificationResult,
    _certification_cohort,
    certify,
    run_sweep,
    sweep_and_certify,
)
from afml.validation import (
    DSRResult,
    PBOResult,
    TargetShufflingResult,
    ValidationResult,
)

pytestmark = pytest.mark.m1

WINNER: Coord = (2, 2)  # interior point of the CUSUM 5×6 grid (full neighborhood)


def _fake_pc(asset: str = "TESTFX") -> AssetPrecompute:
    """Minimal precompute stand-in — ``run_trial`` is stubbed, so bars are unused."""
    return AssetPrecompute(
        asset=asset,
        bars=pl.DataFrame({"close": [1.0, 1.1, 1.2]}),
        bar_type="time",
        bar_parameter="60m",
        bar_jarque_bera=12.34,
        n_bars=3,
        regime_name="day",
        bar_hours=1.5,
        vertical_bars=16,
        target_bar_count=3,
        ffd_d=0.3,
        ffd_window=10,
        ffd_adf_pvalue=0.01,
    )


def _trial(coord: Coord, *, objective: float | None, valid: bool) -> TrialResult:
    return TrialResult(
        family="cusum",
        coord=coord,
        config={"vol_span": 50.0, "threshold_mult": 1.5},
        status="completed",
        n_events=900,
        objective=objective,
        valid=valid,
        brier_calibrated=0.20 if valid else None,
        brier_naive=0.25 if valid else None,
        experiment_id=uuid4(),
        detail="stub",
    )


def _plateau_surface_stub() -> dict[Coord, TrialResult]:
    """A 3×3 high block around WINNER, everything else invalid (-inf)."""
    grid = get_family_grid("cusum")
    block = {(i, j) for i in (1, 2, 3) for j in (1, 2, 3)}
    out: dict[Coord, TrialResult] = {}
    for coord in grid.coords():
        if coord in block:
            out[coord] = _trial(coord, objective=1.0, valid=True)
        else:
            out[coord] = _trial(coord, objective=None, valid=False)
    return out


def _passing_validation() -> ValidationResult:
    return ValidationResult(
        pbo=PBOResult(pbo=0.0, n_splits=5, n_strategies=4, logits=np.zeros(1, dtype=np.float64)),
        dsr=DSRResult(
            dsr=0.99,
            sharpe_observed=1.5,
            expected_max_sharpe=0.2,
            n_trials=30,
            n_observations=900,
            quarantined=False,
        ),
        target_shuffling=TargetShufflingResult(
            brier_real=0.20,
            brier_shuffled=np.array([0.25, 0.26, 0.24], dtype=np.float64),
            pvalue=0.001,
            n_shuffles=30,
        ),
        incumbent_idx=0,
    )


# --------------------------------------------------------------------- cohort


def test_certification_cohort_size_and_freshness() -> None:
    depths = (3, 5, None)
    factories = _certification_cohort(
        n_estimators=10, depths=depths, min_samples_leaf=2, include_xgboost=True, random_state=0
    )
    assert len(factories) == len(depths) + 1  # + XGBoost
    first = factories[0]
    assert first() is not first()  # each call yields a fresh estimator


def test_certification_cohort_without_xgboost() -> None:
    depths = (3, 5, 8, None)
    factories = _certification_cohort(
        n_estimators=10, depths=depths, min_samples_leaf=2, include_xgboost=False, random_state=0
    )
    assert len(factories) == len(depths)


def test_certification_cohort_members_fit_and_predict() -> None:
    factories = _certification_cohort(
        n_estimators=8, depths=(3, None), min_samples_leaf=1, include_xgboost=True, random_state=0
    )
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(np.int64)
    for factory in factories:
        est = factory()
        est.fit(X, y)
        proba = est.predict_proba(X)
        assert proba.shape == (40, 2)


# ---------------------------------------------------------------------- sweep


def test_run_sweep_selects_interior_plateau(monkeypatch: pytest.MonkeyPatch) -> None:
    canned = _plateau_surface_stub()

    def fake_run_trial(pc: AssetPrecompute, family: str, coord: Coord, **_: Any) -> TrialResult:
        return canned[coord]

    monkeypatch.setattr(sweep_mod, "run_trial", fake_run_trial)
    registry = AlphaRegistryRepository("sqlite:///:memory:", wal_mode=False)
    registry.create_all()

    result = run_sweep(_fake_pc(), "cusum", registry=registry)

    assert result.plateau.selected == WINNER
    assert result.winner_trial is not None
    assert result.winner_trial.coord == WINNER
    assert result.surface[WINNER] == pytest.approx(1.0)
    assert result.n_valid == 9  # the 3×3 block


def test_run_sweep_no_plateau_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def all_invalid(pc: AssetPrecompute, family: str, coord: Coord, **_: Any) -> TrialResult:
        return _trial(coord, objective=None, valid=False)

    monkeypatch.setattr(sweep_mod, "run_trial", all_invalid)
    registry = AlphaRegistryRepository("sqlite:///:memory:", wal_mode=False)
    registry.create_all()

    result = run_sweep(_fake_pc(), "cusum", registry=registry)

    assert result.plateau.selected is None
    assert result.winner_trial is None
    assert result.n_valid == 0


# ------------------------------------------------------------------ certify


def test_certify_insufficient_events(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_data(*_: Any, **__: Any) -> tuple[None, int, str]:
        return None, 42, "too few events (42 < 500)"

    monkeypatch.setattr(sweep_mod, "build_event_dataset", no_data)
    registry = AlphaRegistryRepository("sqlite:///:memory:", wal_mode=False)
    registry.create_all()

    cert = certify(_fake_pc(), "cusum", WINNER, registry=registry, experiment_id=None)

    assert cert.status == STATUS_INSUFFICIENT_EVENTS
    assert cert.validation is None
    assert cert.passed is False
    assert cert.n_events == 42


def test_sweep_and_certify_skips_certification_without_plateau(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def all_invalid(pc: AssetPrecompute, family: str, coord: Coord, **_: Any) -> TrialResult:
        return _trial(coord, objective=None, valid=False)

    monkeypatch.setattr(sweep_mod, "run_trial", all_invalid)
    registry = AlphaRegistryRepository("sqlite:///:memory:", wal_mode=False)
    registry.create_all()

    out = sweep_and_certify(_fake_pc(), "cusum", registry=registry)

    assert out.sweep.plateau.selected is None
    assert out.certification is None


# --------------------------------------------------------- result properties


def test_certification_passed_property_true() -> None:
    cert = CertificationResult(
        asset="TESTFX",
        family="cusum",
        coord=WINNER,
        experiment_id=uuid4(),
        n_events=900,
        surviving_features=("roll_measure",),
        n_trials=30,
        status=STATUS_CERTIFIED,
        validation=_passing_validation(),
        detail="ok",
    )
    assert cert.passed is True


def test_certification_passed_false_when_rejected() -> None:
    cert = CertificationResult(
        asset="TESTFX",
        family="cusum",
        coord=WINNER,
        experiment_id=uuid4(),
        n_events=900,
        surviving_features=(),
        n_trials=30,
        status=STATUS_REJECTED,
        validation=_passing_validation(),  # validation passes, but status says rejected
        detail="rejected",
    )
    assert cert.passed is False
