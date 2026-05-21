"""Per-config trial harness (Ops M1.4) — one hyperparameter config → one trial.

Wires the validated phases for a single ``(asset, family, coord)``:

    alpha.detect → triple-barrier → features → align (V2 burn-in) →
    select_features (ONC + Clustered MDA) → train_brain_two (collect OOS) →
    s(g) = median PWF OOS Sharpe

and **logs every config as an Alpha-Registry trial** (drives the DSR ``K``
count): a ``FAILED_AT_MDA`` row when Phase 4's circuit breaker fires, otherwise
a ``completed`` row with the meta-model loss. Dedup makes the sweep resumable —
a config already in the registry is skipped.

Validity (for the plateau surface, not for logging): a config contributes its
``s(g)`` only if it cleared **events ≥ 500** *and* Brain-2 **beat the naive
baseline** (calibrated Brier < naive Brier). Recall is intentionally absent —
it is undefined out-of-sample (no future oracle); it stays a Phase-2 synthetic
DoD. Invalid configs map to ``-inf`` on the surface (still logged as trials).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID

import numpy as np
import numpy.typing as npt
import polars as pl

from afml.core.registry import AlphaRegistryRepository, DuplicateHypothesisError
from afml.data import align_labels_to_features
from afml.features import compute_features
from afml.labeling import (
    BollingerMeanReversion,
    DonchianBreakout,
    PrimaryAlpha,
    SymmetricCUSUM,
    apply_triple_barrier,
)
from afml.modeling import train_brain_two
from afml.research.grids import DEFAULT_SL_MULT, get_family_grid
from afml.research.objective import oos_strategy_sharpe_per_fold
from afml.research.plateau import Coord
from afml.research.precompute import AssetPrecompute
from afml.selection import select_features

MIN_EVENTS: int = 500
DEFAULT_MAX_EVENTS: int = 6000  # surface safety cap (systematic thinning); 0 disables
DEFAULT_TB_VOL_SPAN: int = 50
DEFAULT_FEATURE_WINDOWS: tuple[int, ...] = (10, 20, 30, 50)
DEFAULT_ENTROPY_WINDOWS: tuple[int, ...] = (50, 100)
_MS_PER_YEAR: float = 365.25 * 24.0 * 3600.0 * 1000.0

# Trial status strings (mirror the registry's EXPERIMENT_STATUS_* + a research-only
# "invalid" for configs that never reached the model, e.g. too few events).
STATUS_COMPLETED: str = "completed"
STATUS_FAILED_AT_MDA: str = "FAILED_AT_MDA"
STATUS_INVALID: str = "invalid"


@dataclass(frozen=True, slots=True)
class TrialDiagnostics:
    """Per-stage observability for one trial — the funnel from events to Sharpe.

    Every field is what a human needs to see *where* a config succeeds or
    degrades: how many events fired, whether labels are balanced, whether the
    holding horizon matches the regime, how many feature clusters survived MDA,
    whether the meta-model beat naive, and the per-fold OOS Sharpe spread (a
    tight cluster is trustworthy; a high median on one lucky fold is not).
    Populated progressively — early-exit paths leave later stages ``None``.
    """

    n_alpha_events: int
    n_events_modeled: int | None = None  # after burn-in alignment + thinning
    label_pos_rate: float | None = None  # P[y=1]
    return_mean: float | None = None
    return_std: float | None = None
    mean_holding_bars: float | None = None
    target_holding_bars: int | None = None  # regime vertical V
    n_features_in: int | None = None
    halted_at_mda: bool | None = None
    n_surviving_features: int | None = None
    surviving_features: tuple[str, ...] = ()
    brier_calibrated: float | None = None
    brier_naive: float | None = None
    fold_sharpes: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class TrialResult:
    """One swept config's outcome."""

    family: str
    coord: Coord
    config: dict[str, float]
    status: str
    n_events: int
    objective: float | None  # s(g); None when not computable
    valid: bool  # events ≥ 500 ∧ beats baseline ∧ objective finite
    brier_calibrated: float | None
    brier_naive: float | None
    experiment_id: UUID | None
    detail: str
    diagnostics: TrialDiagnostics | None = None  # per-stage observability

    @property
    def surface_score(self) -> float:
        """Value fed to the plateau selector (``-inf`` when invalid)."""
        if self.valid and self.objective is not None and np.isfinite(self.objective):
            return self.objective
        return float("-inf")


def _build_alpha(family: str, config: dict[str, float]) -> PrimaryAlpha:
    if family == "cusum":
        return SymmetricCUSUM(
            vol_span=int(config["vol_span"]),
            threshold_multiplier=float(config["threshold_mult"]),
        )
    if family == "bollinger":
        return BollingerMeanReversion(
            sma_span=int(config["window"]),
            num_std=float(config["num_std"]),
        )
    if family == "donchian":
        return DonchianBreakout(window=int(config["window"]))
    raise KeyError(f"unknown alpha family {family!r}")


def _side_sign(side_col: pl.Series) -> npt.NDArray[np.int64]:
    """Map a 'side' column ('long'/'short' or +1/-1) to ±1."""
    raw = side_col.to_numpy()
    if raw.dtype.kind in ("U", "S", "O"):
        return np.asarray(np.where(raw == "long", 1, -1), dtype=np.int64)
    return np.asarray(np.sign(raw.astype(np.float64)), dtype=np.int64)


def _periods_per_year(event_ts: pl.Series) -> float:
    ts = event_ts.cast(pl.Int64).to_numpy().astype(np.float64)
    if ts.size < 2:
        return 252.0
    span_ms = float(ts.max() - ts.min())
    if span_ms <= 0.0:
        return 252.0
    return max(ts.size / (span_ms / _MS_PER_YEAR), 1.0)


@dataclass(frozen=True, slots=True)
class EventDataset:
    """Config-specific modelling data (events → labels → features → aligned arrays)."""

    features_aligned: pl.DataFrame
    y: npt.NDArray[np.int64]
    t0: npt.NDArray[np.int64]
    t1: npt.NDArray[np.int64]
    return_pct: npt.NDArray[np.float64]
    side_sign: npt.NDArray[np.int64]
    n_alpha_events: int
    periods_per_year: float


def build_event_dataset(
    pc: AssetPrecompute,
    family: str,
    config: dict[str, float],
    *,
    tb_vol_span: int = DEFAULT_TB_VOL_SPAN,
    feature_windows: tuple[int, ...] = DEFAULT_FEATURE_WINDOWS,
    entropy_windows: tuple[int, ...] = DEFAULT_ENTROPY_WINDOWS,
    min_events: int = MIN_EVENTS,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> tuple[EventDataset | None, int, str]:
    """Events → triple-barrier → features → V2-aligned arrays for one config.

    Returns ``(dataset_or_None, n_alpha_events, detail)``. ``None`` ⇒ too few
    events (pre- or post-burn-in). Shared by ``run_trial`` (surface) and the
    M1.6 ``certify`` path so both consume identical data prep.
    """
    profit_take_mult = float(config.get("pt_mult", 2.0))
    events = _build_alpha(family, config).detect(pc.bars)
    n_alpha = events.height
    if n_alpha < min_events:
        return None, n_alpha, f"too few events ({n_alpha} < {min_events})"

    labels = apply_triple_barrier(
        pc.bars,
        events,
        vol_span=tb_vol_span,
        profit_take_mult=profit_take_mult,
        stop_loss_mult=DEFAULT_SL_MULT,
        vertical_barrier_bars=pc.vertical_bars,
    )
    features = compute_features(
        pc.bars, events, windows=feature_windows, windows_entropy=entropy_windows
    )
    aligned_labels, _report = align_labels_to_features(labels.df, features)
    aligned_labels = aligned_labels.sort("event_timestamp")
    if aligned_labels.height < min_events:
        return None, n_alpha, "too few events after burn-in alignment"

    if max_events > 0 and aligned_labels.height > max_events:
        step = int(np.ceil(aligned_labels.height / max_events))
        aligned_labels = aligned_labels.gather_every(step)

    features_aligned = features.join(
        aligned_labels.select("event_timestamp").rename({"event_timestamp": "timestamp"}),
        on="timestamp",
        how="inner",
    ).sort("timestamp")
    dataset = EventDataset(
        features_aligned=features_aligned,
        y=aligned_labels["label"].to_numpy().astype(np.int64),
        t0=aligned_labels["event_timestamp"].cast(pl.Int64).to_numpy().astype(np.int64),
        t1=aligned_labels["exit_timestamp"].cast(pl.Int64).to_numpy().astype(np.int64),
        return_pct=aligned_labels["return_pct"].to_numpy().astype(np.float64),
        side_sign=_side_sign(aligned_labels["side"]),
        n_alpha_events=n_alpha,
        periods_per_year=_periods_per_year(aligned_labels["event_timestamp"]),
    )
    return dataset, n_alpha, "ok"


def _dataset_diagnostics(dataset: EventDataset, pc: AssetPrecompute) -> TrialDiagnostics:
    """Stage 2-3 metrics; the FAILED_AT_MDA / completed paths extend it via ``replace``."""
    bar_ms = pc.bar_hours * 3_600_000.0
    hold_ms = (dataset.t1 - dataset.t0).astype(np.float64)
    y = dataset.y
    r = dataset.return_pct
    n_feat = len([c for c in dataset.features_aligned.columns if c != "timestamp"])
    return TrialDiagnostics(
        n_alpha_events=dataset.n_alpha_events,
        n_events_modeled=int(y.size),
        label_pos_rate=float(y.mean()) if y.size else None,
        return_mean=float(r.mean()) if r.size else None,
        return_std=float(r.std(ddof=1)) if r.size > 1 else None,
        mean_holding_bars=(float(hold_ms.mean() / bar_ms) if bar_ms > 0 and hold_ms.size else None),
        target_holding_bars=pc.vertical_bars,
        n_features_in=n_feat,
    )


def run_trial(
    pc: AssetPrecompute,
    family: str,
    coord: Coord,
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
    compare_with_xgboost: bool = False,
    estimator: str = "rf",
    random_state: int = 0,
) -> TrialResult:
    """Run one config end-to-end and log it to the Alpha Registry.

    Economy defaults for the cheap surface pre-pass: ``compare_with_xgboost=False``,
    a capped forest (``n_estimators``), and ``max_events`` systematic thinning of
    the busiest configs. Certification (M1.6) re-fits the survivor with full
    settings. The triple-barrier vertical comes from the regime (``pc.vertical_bars``).
    """
    grid = get_family_grid(family)
    config = grid.config(coord)

    def _invalid(n: int, detail: str) -> TrialResult:
        return TrialResult(
            family,
            coord,
            config,
            STATUS_INVALID,
            n,
            None,
            False,
            None,
            None,
            None,
            detail,
            diagnostics=TrialDiagnostics(n_alpha_events=n),
        )

    # --- Phases 2-3: events → labels → features → V2-aligned arrays ----------
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
        return _invalid(n_alpha, detail)

    features_aligned = dataset.features_aligned
    y, t0, t1 = dataset.y, dataset.t0, dataset.t1
    hyperparameter_vector: dict[str, Any] = {"family": family, **config}

    # --- Phase 4: selection (registry logging owned by the harness) ----------
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
        exp_id = _record(
            registry, "failed", agent_version, pc.asset, family, hyperparameter_vector, n_alpha
        )
        diag = replace(
            _dataset_diagnostics(dataset, pc), halted_at_mda=True, n_surviving_features=0
        )
        return TrialResult(
            family,
            coord,
            config,
            STATUS_FAILED_AT_MDA,
            n_alpha,
            None,
            False,
            None,
            None,
            exp_id,
            "empty MDA survivors",
            diagnostics=diag,
        )

    # --- Phase 5: Brain 2 (collect OOS for the objective) --------------------
    x_selected = features_aligned.select(selection.surviving_features).to_numpy().astype(np.float64)
    brain2 = train_brain_two(
        x_selected,
        y,
        t0,
        t1,
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        n_estimators=n_estimators,
        compare_with_xgboost=compare_with_xgboost,
        estimator=estimator,
        random_state=random_state,
        collect_oos_predictions=True,
    )
    brier_cal = brain2.mean_calibrated_brier
    brier_naive = brain2.mean_naive_brier
    beats_baseline = bool(brier_cal < brier_naive)

    # --- Objective s(g) (+ per-fold spread for diagnostics) ------------------
    fold_sharpes = oos_strategy_sharpe_per_fold(
        brain2,
        dataset.return_pct,
        dataset.side_sign,
        periods_per_year=dataset.periods_per_year,
    )
    objective = float(np.median(fold_sharpes)) if fold_sharpes else None

    valid = bool(beats_baseline and objective is not None and np.isfinite(objective))
    exp_id = _record(
        registry,
        "completed",
        agent_version,
        pc.asset,
        family,
        hyperparameter_vector,
        n_alpha,
        brain_2_log_loss=brier_cal,
    )
    diag = replace(
        _dataset_diagnostics(dataset, pc),
        halted_at_mda=False,
        n_surviving_features=len(selection.surviving_features),
        surviving_features=tuple(selection.surviving_features),
        brier_calibrated=brier_cal,
        brier_naive=brier_naive,
        fold_sharpes=tuple(fold_sharpes),
    )
    return TrialResult(
        family,
        coord,
        config,
        STATUS_COMPLETED,
        n_alpha,
        objective,
        valid,
        brier_cal,
        brier_naive,
        exp_id,
        "ok" if valid else "completed but invalid for surface",
        diagnostics=diag,
    )


def _record(
    registry: AlphaRegistryRepository,
    kind: str,
    agent_version: str,
    asset: str,
    family: str,
    hyperparameter_vector: dict[str, Any],
    num_events: int,
    *,
    brain_2_log_loss: float | None = None,
) -> UUID | None:
    """Log a trial; swallow ``DuplicateHypothesisError`` so the sweep is resumable."""
    try:
        if kind == "failed":
            return registry.record_failed_mda(
                agent_version=agent_version,
                asset=asset,
                algorithmic_family=family,
                hyperparameter_vector=hyperparameter_vector,
                num_events_triggered=num_events,
            )
        return registry.record_experiment(
            agent_version=agent_version,
            asset=asset,
            algorithmic_family=family,
            hyperparameter_vector=hyperparameter_vector,
            num_events_triggered=num_events,
            brain_2_log_loss=brain_2_log_loss,
        )
    except DuplicateHypothesisError:
        return None
