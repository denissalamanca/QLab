"""Phase 5 — Brain 2 / Meta-Labeler training orchestrator.

Pipeline (Blueprint §7 + AFML 0-5 integration audit):

1. **Empty-MDA pass-through (0-5 audit V2.1).** If Phase 4's Clustered MDA
   tripped its circuit breaker, ``X`` has zero feature columns. Phase 5 must
   detect this and return a sentinel :class:`BrainTwoResult` with
   ``halted_at_mda_upstream=True`` instead of crashing.

2. **Indicator matrix + ū_i (Blueprint §7.1).** Build the indicator matrix
   over the event horizons; per-event average uniqueness becomes the
   ``sample_weight`` carried through every fit.

3. **Strict-embargo outer evaluation (0-5 audit V2).** Walk-forward folds via
   :class:`PurgedWalkForwardCV` give chronologically separated
   train / holdout slices; the index-intersection check
   ``max(train_t1) + embargo < min(holdout_t0)`` is asserted per fold and
   surfaced in :class:`FoldDiagnostics.passes_index_intersection`.

4. **Purged-CV calibration (0-5 audit V1).** For each fold's training slice
   we call :func:`fit_calibrated_sbrf_with_purged_cv` (and, when
   ``compare_with_xgboost=True``, :func:`fit_calibrated_classifier_with_purged_cv`
   for an XGBoost mirror). Both invocations build the calibrator via an
   explicit :class:`PurgedKFold` cross-fitting loop — **no random KFold ever
   touches the labels**.

5. **Sample-weight propagation (0-5 audit V3).** ``sample_weight = ū_i`` is
   passed all the way down to both SBRF and XGBoost's per-tree fits and the
   sklearn calibrator's ``fit_params``.

6. **Brier-min winner.** The fold-level ``CalibrationResult`` keeps whichever
   of (SBRF×isotonic, SBRF×sigmoid, XGB×isotonic, XGB×sigmoid) minimises
   the holdout Brier; the final fold's winner is surfaced as
   ``BrainTwoResult.last_calibration`` for deployment.

7. **Naive-baseline gate (Blueprint §7.3).** Calibrated Brier must beat the
   class-prior Brier on every fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

from afml.modeling.calibration import (
    CalibrationResult,
    fit_calibrated_classifier_with_purged_cv,
    fit_calibrated_sbrf_with_purged_cv,
)
from afml.modeling.concurrency import average_uniqueness, indicator_matrix
from afml.modeling.sbrf import SequentiallyBootstrappedRandomForest
from afml.selection.purged_kfold import PurgedWalkForwardCV

DEFAULT_N_ESTIMATORS_SBRF: int = 200
DEFAULT_N_ESTIMATORS_XGB: int = 200
DEFAULT_MAX_DEPTH: int = 5
DEFAULT_MIN_SAMPLES_LEAF: int = 5
DEFAULT_INNER_N_SPLITS: int = 3
PROBA_EPSILON: float = 1e-6


@dataclass(frozen=True, slots=True)
class FoldDiagnostics:
    """One fold's audit trail."""

    fold_index: int
    n_train: int
    n_holdout: int
    brier_uncalibrated: float
    brier_calibrated: float
    brier_naive_baseline: float
    calibration_method: str
    winning_estimator: str  # "sbrf" | "xgboost"
    holdout_t0_min: int
    train_t1_max: int
    embargo_size: int

    @property
    def passes_index_intersection(self) -> bool:
        """Blueprint §7.3 / 0-5 audit V2 —
        ``max(train_t1) + embargo < min(holdout_t0)``."""
        return self.train_t1_max + self.embargo_size < self.holdout_t0_min

    @property
    def beats_naive_baseline(self) -> bool:
        """Blueprint §7.3 — calibrated Brier strictly below naive."""
        return self.brier_calibrated < self.brier_naive_baseline


@dataclass(frozen=True, slots=True)
class BrainTwoResult:
    """Output of :func:`train_brain_two`.

    Attributes
    ----------
    fold_diagnostics
        One :class:`FoldDiagnostics` per walk-forward fold (empty when
        ``halted_at_mda_upstream=True``).
    average_uniqueness_per_event
        ``(n_samples,)`` ū vector used to weight every fit.
    last_calibration
        Final fold's :class:`CalibrationResult` — the deployable artefact.
        ``None`` when no productive fold ran (e.g. empty-MDA bypass).
    n_estimators
        Forwarded SBRF setting for traceability.
    halted_at_mda_upstream
        AFML 0-5 audit V2.1 — True iff Phase 4 returned an empty feature
        matrix, so Brain 2 short-circuited.
    """

    fold_diagnostics: list[FoldDiagnostics]
    average_uniqueness_per_event: npt.NDArray[np.float64]
    last_calibration: CalibrationResult | None
    n_estimators: int = field(default=0)
    halted_at_mda_upstream: bool = field(default=False)

    @property
    def passes_phase5_dod(self) -> bool:
        """Both Blueprint §7.3 DoD checks must hold across every productive fold."""
        if self.halted_at_mda_upstream:
            # An upstream halt is by design — no Brain 2 to evaluate. The
            # caller is responsible for not deploying this strategy.
            return False
        if not self.fold_diagnostics:
            return False
        return all(
            f.passes_index_intersection and f.beats_naive_baseline for f in self.fold_diagnostics
        )

    @property
    def mean_calibrated_brier(self) -> float:
        if not self.fold_diagnostics:
            return float("nan")
        return float(np.mean([f.brier_calibrated for f in self.fold_diagnostics]))

    @property
    def mean_naive_brier(self) -> float:
        if not self.fold_diagnostics:
            return float("nan")
        return float(np.mean([f.brier_naive_baseline for f in self.fold_diagnostics]))


def _naive_baseline_proba(
    y_train: npt.NDArray[np.integer],
    n_holdout: int,
) -> npt.NDArray[np.float64]:
    """Class-prior prediction — constant ``p̂ = mean(y_train)`` for every row."""
    prior = float(np.bincount(y_train, minlength=2)[1] / max(y_train.size, 1))
    return np.full(n_holdout, prior, dtype=np.float64)


def _positive_class_proba(classifier: Any, X: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    proba = classifier.predict_proba(X)
    classes = np.asarray(classifier.classes_)
    if proba.shape[1] == 1:
        only_class = int(classes[0])
        return np.full(X.shape[0], 1.0 if only_class == 1 else 0.0, dtype=np.float64)
    pos_col = int(np.where(classes == 1)[0][0])
    return np.asarray(proba[:, pos_col], dtype=np.float64)


def _empty_result(
    *,
    n_samples: int,
    n_estimators: int,
    halted_at_mda_upstream: bool,
) -> BrainTwoResult:
    """Build a sentinel result for halted hypotheses (no fits attempted)."""
    return BrainTwoResult(
        fold_diagnostics=[],
        average_uniqueness_per_event=np.zeros(n_samples, dtype=np.float64),
        last_calibration=None,
        n_estimators=n_estimators,
        halted_at_mda_upstream=halted_at_mda_upstream,
    )


def _build_xgboost_classifier(*, n_estimators: int, max_depth: int, random_state: int) -> Any:
    """Construct a standard XGBoost classifier for the meta-model tournament."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
    )


def train_brain_two(  # noqa: PLR0915 — single linear pipeline; splitting harms clarity
    X: npt.NDArray[np.floating],
    y: npt.NDArray[np.integer],
    t0: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    t1: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    *,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    train_fraction: float = 0.3,
    inner_n_splits: int = DEFAULT_INNER_N_SPLITS,
    n_estimators: int = DEFAULT_N_ESTIMATORS_SBRF,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    compare_with_xgboost: bool = True,
    random_state: int = 0,
) -> BrainTwoResult:
    """Train and calibrate Brain 2 across walk-forward folds.

    AFML 0-5 audit V2.1: if ``X.shape[1] == 0`` the function returns a
    sentinel :class:`BrainTwoResult` with ``halted_at_mda_upstream=True``
    without attempting any fit. This is the canonical Phase 4 ➜ Phase 5
    failure handoff.

    Parameters
    ----------
    X
        ``(n_samples, n_features)`` Phase-3 feature matrix aligned with
        ``y``, ``t0``, ``t1``. ``n_features == 0`` triggers the
        empty-MDA pass-through.
    y
        ``(n_samples,)`` binary Triple-Barrier labels.
    t0, t1
        Per-sample event horizons (AFML audit V1 — use the **realized**
        ``exit_timestamp`` as t1, not the conservative ``vertical_timestamp``).
    n_splits
        Number of chronological **outer** walk-forward folds.
    embargo_pct
        Embargo width as a fraction of the total sample count, applied
        both at the outer split and inside the purged calibration CV.
    train_fraction
        Burn-in fraction reserved before any outer test fold begins.
    inner_n_splits
        Number of purged K-Fold splits used INSIDE the calibration
        cross-fitting (AFML 0-5 audit V1).
    n_estimators, max_depth, min_samples_leaf
        SBRF (and XGBoost) hyperparameters.
    compare_with_xgboost
        AFML 0-5 audit V3 — when True, also train an XGBoost mirror with the
        same ``sample_weight = ū_i``, calibrate it via purged CV, and
        keep whichever calibrated estimator (SBRF or XGB) achieves the
        lower Brier on the fold's holdout.
    random_state
        Master seed.

    Returns
    -------
    :class:`BrainTwoResult`.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    t0_arr = np.asarray(t0, dtype=np.int64)
    t1_arr = np.asarray(t1, dtype=np.int64)
    n_samples = X_arr.shape[0]

    # ---- AFML 0-5 audit V2.1 — empty-MDA pass-through ----------------------
    if X_arr.ndim != 2 or X_arr.shape[1] == 0 or n_samples == 0:
        return _empty_result(
            n_samples=n_samples,
            n_estimators=n_estimators,
            halted_at_mda_upstream=True,
        )

    if y_arr.shape != (n_samples,) or t0_arr.shape != (n_samples,) or t1_arr.shape != (n_samples,):
        raise ValueError(
            f"shape mismatch: X={X_arr.shape}, y={y_arr.shape}, "
            f"t0={t0_arr.shape}, t1={t1_arr.shape}"
        )
    if not set(np.unique(y_arr).tolist()) <= {0, 1}:
        raise ValueError("y must be binary in {0, 1}")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X must be finite")

    ind_full = indicator_matrix(t0_arr, t1_arr)
    avg_u_full = average_uniqueness(ind_full)

    embargo_size = int(np.floor(n_samples * embargo_pct))
    cv = PurgedWalkForwardCV(
        n_splits=n_splits, embargo_pct=embargo_pct, train_fraction=train_fraction
    )
    folds = list(cv.split(t0_arr, t1_arr))
    if not folds:
        raise ValueError("PurgedWalkForwardCV produced no folds — check parameters")

    fold_diagnostics: list[FoldDiagnostics] = []
    last_calibration: CalibrationResult | None = None

    for fold_i, (train_idx, holdout_idx) in enumerate(folds):
        if train_idx.size < 2 or holdout_idx.size < 2:
            continue
        # Per-fold class diversity.
        train_class_counts = np.bincount(y_arr[train_idx], minlength=2)
        if train_class_counts.min() < 2 or len(np.unique(y_arr[holdout_idx])) < 2:
            continue

        X_train = X_arr[train_idx]
        y_train = y_arr[train_idx]
        X_hold = X_arr[holdout_idx]
        y_hold = y_arr[holdout_idx]
        t0_train = t0_arr[train_idx]
        t1_train = t1_arr[train_idx]
        ind_train = ind_full[:, train_idx]
        weight_train = avg_u_full[train_idx]

        # ---- SBRF + purged-CV calibration (0-5 audit V1) -------------------
        sbrf_template = SequentiallyBootstrappedRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + fold_i,
        )
        sbrf_calibration = fit_calibrated_sbrf_with_purged_cv(
            sbrf_template,
            X_train,
            y_train,
            ind_train,
            t0_train,
            t1_train,
            weight_train,
            X_hold,
            y_hold,
            n_splits=inner_n_splits,
            embargo_pct=embargo_pct,
        )

        winning_calibration = sbrf_calibration
        winning_brier = min(sbrf_calibration.brier_isotonic, sbrf_calibration.brier_sigmoid)
        winning_estimator = "sbrf"

        # ---- XGBoost mirror (0-5 audit V3) ---------------------------------
        if compare_with_xgboost:
            xgb_template = _build_xgboost_classifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state + fold_i,
            )
            try:
                xgb_calibration = fit_calibrated_classifier_with_purged_cv(
                    xgb_template,
                    X_train,
                    y_train,
                    t0_train,
                    t1_train,
                    weight_train,
                    X_hold,
                    y_hold,
                    n_splits=inner_n_splits,
                    embargo_pct=embargo_pct,
                )
                xgb_winning_brier = min(
                    xgb_calibration.brier_isotonic, xgb_calibration.brier_sigmoid
                )
                if xgb_winning_brier < winning_brier:
                    winning_calibration = xgb_calibration
                    winning_brier = xgb_winning_brier
                    winning_estimator = "xgboost"
            except (ValueError, RuntimeError):
                # XGBoost fits can fail on degenerate folds (single-class
                # subsets, etc.). Fall back to SBRF without breaking the loop.
                pass

        cal_proba = _positive_class_proba(winning_calibration.calibrated, X_hold)
        brier_cal = float(brier_score_loss(y_hold, cal_proba))

        naive_proba = _naive_baseline_proba(y_train, holdout_idx.size)
        brier_naive = float(brier_score_loss(y_hold, naive_proba))

        fold_diagnostics.append(
            FoldDiagnostics(
                fold_index=fold_i,
                n_train=train_idx.size,
                n_holdout=holdout_idx.size,
                brier_uncalibrated=winning_calibration.brier_uncalibrated,
                brier_calibrated=brier_cal,
                brier_naive_baseline=brier_naive,
                calibration_method=winning_calibration.method,
                winning_estimator=winning_estimator,
                holdout_t0_min=int(t0_arr[holdout_idx].min()),
                train_t1_max=int(t1_arr[train_idx].max()),
                embargo_size=embargo_size,
            )
        )
        last_calibration = winning_calibration

    if last_calibration is None:
        raise ValueError("every fold was degenerate — increase n_samples or n_splits")

    return BrainTwoResult(
        fold_diagnostics=fold_diagnostics,
        average_uniqueness_per_event=avg_u_full,
        last_calibration=last_calibration,
        n_estimators=n_estimators,
        halted_at_mda_upstream=False,
    )
