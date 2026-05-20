"""Sweep + two-stage certification (Ops M1.6) — the research run, end-to-end.

Stage 1 — **surface** (:func:`run_sweep`). Score every config of a family grid
with the cheap RF surface harness (:func:`afml.research.run_trial`), build the
``{coord: s(g)}`` surface, and pick the robust centre with the
Neighborhood-Minimax plateau selector (:func:`afml.research.select_plateau`).
Every config is logged to the Alpha Registry as a trial (drives the DSR ``K``
count); the sweep is resumable through registry dedup.

Stage 2 — **certification** (:func:`certify`). Re-derive the winner's data,
reselect features, and run the Phase 6 gate (:func:`validate_strategy`) over a
**fast-classifier CPCV cohort** — a RandomForest complexity ladder (+ optional
XGBoost). Why not the deployable SBRF here? CPCV refits the whole cohort across
``C(N,k)`` combinations and the target-shuffling loop refits the incumbent
``n_shuffles`` times; the sequential-bootstrap SBRF is ``O(n²)`` per fit and
would make certification intractable. The depth ladder is also the *right*
cohort for PBO — model complexity is exactly the overfitting axis PBO measures.
The certified ``(pbo, dsr)`` is attached to the winner's registry row; M2 trains
the deployable weighted SBRF on the certified config.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any
from uuid import UUID

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from afml.core.registry import AlphaRegistryRepository
from afml.research.grids import get_family_grid
from afml.research.harness import (
    DEFAULT_ENTROPY_WINDOWS,
    DEFAULT_FEATURE_WINDOWS,
    DEFAULT_MAX_EVENTS,
    DEFAULT_TB_VOL_SPAN,
    MIN_EVENTS,
    TrialResult,
    build_event_dataset,
    run_trial,
)
from afml.research.plateau import Coord, PlateauResult, select_plateau
from afml.research.precompute import AssetPrecompute
from afml.selection import select_features
from afml.validation import (
    DataLeakageError,
    ValidationResult,
    count_cohort_trials,
    validate_strategy,
)

#: Certification cohort — a RandomForest complexity ladder (the PBO overfitting
#: axis). ``None`` ⇒ unbounded depth (the most overfit member).
DEFAULT_COHORT_DEPTHS: tuple[int | None, ...] = (3, 5, 8, None)
DEFAULT_COHORT_N_ESTIMATORS: int = 200
DEFAULT_COHORT_MIN_SAMPLES_LEAF: int = 5

# Certification status strings.
STATUS_CERTIFIED: str = "certified"
STATUS_REJECTED: str = "rejected"
STATUS_HALTED_MDA: str = "halted_at_mda"
STATUS_INSUFFICIENT_EVENTS: str = "insufficient_events"
STATUS_DATA_LEAKAGE: str = "data_leakage"
STATUS_DEGENERATE: str = "degenerate_cpcv"


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Stage-1 outcome for one ``(asset, family)`` grid."""

    asset: str
    family: str
    trials: tuple[TrialResult, ...]
    surface: dict[Coord, float]
    plateau: PlateauResult
    winner_trial: TrialResult | None

    @property
    def n_valid(self) -> int:
        """Configs that contributed a finite score to the plateau surface."""
        return sum(1 for t in self.trials if t.valid)


@dataclass(frozen=True, slots=True)
class CertificationResult:
    """Stage-2 Phase 6 outcome for the plateau winner."""

    asset: str
    family: str
    coord: Coord
    experiment_id: UUID | None
    n_events: int
    surviving_features: tuple[str, ...]
    n_trials: int
    status: str
    validation: ValidationResult | None
    detail: str

    @property
    def passed(self) -> bool:
        """True only on a clean Phase 6 pass (PBO < thr ∧ DSR > thr ∧ no leak)."""
        return (
            self.status == STATUS_CERTIFIED
            and self.validation is not None
            and self.validation.passes_phase6_dod
        )


@dataclass(frozen=True, slots=True)
class SweepCertification:
    """The full M1.6 unit: surface sweep + (when a plateau exists) certification."""

    sweep: SweepResult
    certification: CertificationResult | None


def run_sweep(
    pc: AssetPrecompute,
    family: str,
    *,
    registry: AlphaRegistryRepository,
    agent_version: str = "research_harness@m1",
    tb_vol_span: int = DEFAULT_TB_VOL_SPAN,
    feature_windows: tuple[int, ...] = DEFAULT_FEATURE_WINDOWS,
    entropy_windows: tuple[int, ...] = DEFAULT_ENTROPY_WINDOWS,
    min_events: int = MIN_EVENTS,
    max_events: int = DEFAULT_MAX_EVENTS,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    n_estimators: int = 100,
    estimator: str = "rf",
    s_floor: float = 0.0,
    plateau_delta: float = 0.1,
    random_state: int = 0,
) -> SweepResult:
    """Score the whole family grid → plateau-select the robust centre.

    Iterates every grid coord through :func:`run_trial` (cheap RF surface),
    assembles the ``{coord: surface_score}`` map, and runs
    :func:`select_plateau`. Returns the trials, the surface, the plateau verdict,
    and the winning :class:`TrialResult` (``None`` when no stable plateau exists).
    """
    grid = get_family_grid(family)
    trials: list[TrialResult] = []
    surface: dict[Coord, float] = {}
    for coord in grid.coords():
        trial = run_trial(
            pc,
            family,
            coord,
            registry=registry,
            agent_version=agent_version,
            tb_vol_span=tb_vol_span,
            feature_windows=feature_windows,
            entropy_windows=entropy_windows,
            min_events=min_events,
            max_events=max_events,
            n_splits=n_splits,
            embargo_pct=embargo_pct,
            n_estimators=n_estimators,
            estimator=estimator,
            random_state=random_state,
        )
        trials.append(trial)
        surface[coord] = trial.surface_score

    plateau = select_plateau(surface, dims=grid.dims, s_floor=s_floor, delta=plateau_delta)
    winner_trial: TrialResult | None = None
    if plateau.selected is not None:
        winner_trial = next((t for t in trials if t.coord == plateau.selected), None)
    return SweepResult(pc.asset, family, tuple(trials), surface, plateau, winner_trial)


def _certification_cohort(
    *,
    n_estimators: int,
    depths: Sequence[int | None],
    min_samples_leaf: int,
    include_xgboost: bool,
    random_state: int,
) -> list[Callable[[], Any]]:
    """A RandomForest complexity ladder (+ optional XGBoost) of fresh-estimator factories."""
    factories: list[Callable[[], Any]] = []
    for depth in depths:
        factories.append(
            partial(
                RandomForestClassifier,
                n_estimators=n_estimators,
                max_depth=depth,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-1,
                random_state=random_state,
            )
        )
    if include_xgboost:
        max_xgb_depth = max((d for d in depths if d is not None), default=6)
        factories.append(
            partial(
                XGBClassifier,
                n_estimators=n_estimators,
                max_depth=max_xgb_depth,
                random_state=random_state,
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            )
        )
    return factories


def certify(
    pc: AssetPrecompute,
    family: str,
    coord: Coord,
    *,
    registry: AlphaRegistryRepository,
    experiment_id: UUID | None,
    tb_vol_span: int = DEFAULT_TB_VOL_SPAN,
    feature_windows: tuple[int, ...] = DEFAULT_FEATURE_WINDOWS,
    entropy_windows: tuple[int, ...] = DEFAULT_ENTROPY_WINDOWS,
    min_events: int = MIN_EVENTS,
    max_events: int = DEFAULT_MAX_EVENTS,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    cohort_n_estimators: int = DEFAULT_COHORT_N_ESTIMATORS,
    cohort_depths: Sequence[int | None] = DEFAULT_COHORT_DEPTHS,
    cohort_min_samples_leaf: int = DEFAULT_COHORT_MIN_SAMPLES_LEAF,
    include_xgboost: bool = True,
    n_groups: int = 6,
    n_test_groups: int = 2,
    n_shuffles: int = 30,
    pbo_threshold: float = 0.05,
    dsr_threshold: float = 0.95,
    random_state: int = 0,
) -> CertificationResult:
    """Run the Phase 6 gate on one config over a fast-classifier CPCV cohort.

    Re-derives the config's modelling data (identical prep to the surface via
    :func:`build_event_dataset`), reselects features, then calls
    :func:`validate_strategy` with a RandomForest depth ladder (+ optional
    XGBoost). On a clean run the ``(pbo, dsr)`` are written back to the winner's
    registry row via :meth:`record_validation`. Failures (too few events, empty
    MDA, degenerate CPCV, detected leakage) yield a typed status — never a crash,
    so a batch sweep keeps going.
    """

    def _early(status: str, n_events: int, detail: str) -> CertificationResult:
        return CertificationResult(
            pc.asset, family, coord, experiment_id, n_events, (), 0, status, None, detail
        )

    grid = get_family_grid(family)
    config = grid.config(coord)
    dataset, n_alpha, detail = build_event_dataset(
        pc,
        family,
        config,
        tb_vol_span=tb_vol_span,
        feature_windows=feature_windows,
        entropy_windows=entropy_windows,
        min_events=min_events,
        max_events=max_events,
    )
    if dataset is None:
        return _early(STATUS_INSUFFICIENT_EVENTS, n_alpha, detail)

    features_aligned = dataset.features_aligned
    y, t0, t1 = dataset.y, dataset.t0, dataset.t1
    selection = select_features(
        features_aligned,
        y,
        t0,
        t1,
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        random_state=random_state,
    )
    if selection.halted_at_mda:
        return _early(STATUS_HALTED_MDA, n_alpha, "empty MDA survivors at certification")

    x_selected = features_aligned.select(selection.surviving_features).to_numpy().astype(np.float64)
    factories = _certification_cohort(
        n_estimators=cohort_n_estimators,
        depths=cohort_depths,
        min_samples_leaf=cohort_min_samples_leaf,
        include_xgboost=include_xgboost,
        random_state=random_state,
    )
    n_trials = count_cohort_trials(registry, asset=pc.asset, algorithmic_family=family)

    try:
        result = validate_strategy(
            factories,
            x_selected,
            y,
            t0,
            t1,
            dataset.return_pct,
            n_trials=n_trials,
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            embargo_pct=embargo_pct,
            pbo_threshold=pbo_threshold,
            dsr_threshold=dsr_threshold,
            n_shuffles=n_shuffles,
            periods_per_year=max(round(dataset.periods_per_year), 1),
            random_state=random_state,
            raise_on_leakage=True,
        )
    except DataLeakageError as exc:
        return _early(STATUS_DATA_LEAKAGE, n_alpha, str(exc))
    except ValueError as exc:
        return _early(STATUS_DEGENERATE, n_alpha, str(exc))

    if experiment_id is not None:
        registry.record_validation(experiment_id, pbo=result.pbo.pbo, dsr=result.dsr.dsr)

    passed = result.passes_phase6_dod
    status = STATUS_CERTIFIED if passed else STATUS_REJECTED
    detail = (
        f"PBO={result.pbo.pbo:.4f} DSR={result.dsr.dsr:.4f} "
        f"K={n_trials} quarantined={result.dsr.quarantined}"
    )
    return CertificationResult(
        pc.asset,
        family,
        coord,
        experiment_id,
        n_alpha,
        tuple(selection.surviving_features),
        n_trials,
        status,
        result,
        detail,
    )


def sweep_and_certify(
    pc: AssetPrecompute,
    family: str,
    *,
    registry: AlphaRegistryRepository,
    agent_version: str = "research_harness@m1",
    tb_vol_span: int = DEFAULT_TB_VOL_SPAN,
    feature_windows: tuple[int, ...] = DEFAULT_FEATURE_WINDOWS,
    entropy_windows: tuple[int, ...] = DEFAULT_ENTROPY_WINDOWS,
    min_events: int = MIN_EVENTS,
    max_events: int = DEFAULT_MAX_EVENTS,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    surface_n_estimators: int = 100,
    s_floor: float = 0.0,
    plateau_delta: float = 0.1,
    cohort_n_estimators: int = DEFAULT_COHORT_N_ESTIMATORS,
    cohort_depths: Sequence[int | None] = DEFAULT_COHORT_DEPTHS,
    include_xgboost: bool = True,
    n_groups: int = 6,
    n_test_groups: int = 2,
    n_shuffles: int = 30,
    pbo_threshold: float = 0.05,
    dsr_threshold: float = 0.95,
    random_state: int = 0,
) -> SweepCertification:
    """End-to-end M1.6 unit: surface sweep then certify the plateau winner.

    Returns the :class:`SweepResult` always; the :class:`CertificationResult` is
    ``None`` when the sweep found no stable plateau (an expected, valid research
    outcome — there is simply no robust configuration to certify).
    """
    sweep = run_sweep(
        pc,
        family,
        registry=registry,
        agent_version=agent_version,
        tb_vol_span=tb_vol_span,
        feature_windows=feature_windows,
        entropy_windows=entropy_windows,
        min_events=min_events,
        max_events=max_events,
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        n_estimators=surface_n_estimators,
        s_floor=s_floor,
        plateau_delta=plateau_delta,
        random_state=random_state,
    )
    if sweep.plateau.selected is None or sweep.winner_trial is None:
        return SweepCertification(sweep, None)

    certification = certify(
        pc,
        family,
        sweep.plateau.selected,
        registry=registry,
        experiment_id=sweep.winner_trial.experiment_id,
        tb_vol_span=tb_vol_span,
        feature_windows=feature_windows,
        entropy_windows=entropy_windows,
        min_events=min_events,
        max_events=max_events,
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        cohort_n_estimators=cohort_n_estimators,
        cohort_depths=cohort_depths,
        include_xgboost=include_xgboost,
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        n_shuffles=n_shuffles,
        pbo_threshold=pbo_threshold,
        dsr_threshold=dsr_threshold,
        random_state=random_state,
    )
    return SweepCertification(sweep, certification)
