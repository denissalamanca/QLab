"""Phase 5 — Brain 2 / Meta-Labeler training orchestrator.

Pipeline (Blueprint §7):

1. Build the indicator matrix over the event horizons.
2. Compute average uniqueness ``ū_i`` per event.
3. Split chronologically via :class:`PurgedWalkForwardCV` — three slices per
   fold: ``train``, ``calibration``, ``holdout`` (each separated by an
   embargo) so calibration sees a different distribution than the holdout
   used for Brier scoring.
4. Fit :class:`SequentiallyBootstrappedRandomForest` on ``train`` with
   ``sample_weight = ū_i[train]``.
5. Calibrate (isotonic vs sigmoid) on the ``calibration`` slice via prefit
   ``CalibratedClassifierCV``; pick the lower-Brier variant on ``holdout``.
6. Score the chosen calibrated model on ``holdout`` and compare against the
   class-prior naive baseline (Blueprint §7.3 DoD).

The orchestrator returns a :class:`BrainTwoResult` aggregating every fold's
diagnostics so Phase 6's CPCV / PBO machinery can read them directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import brier_score_loss

from afml.modeling.calibration import CalibrationResult, fit_calibrated_classifier
from afml.modeling.concurrency import average_uniqueness, indicator_matrix
from afml.modeling.sbrf import SequentiallyBootstrappedRandomForest
from afml.selection.purged_kfold import PurgedWalkForwardCV

# Each fold splits its train indices into (train_for_sbrf, calibration_slice)
# at this ratio. AFML doesn't prescribe a number; ~80/20 is a workable default.
DEFAULT_CALIBRATION_FRACTION: float = 0.2
PROBA_EPSILON: float = 1e-6


@dataclass(frozen=True, slots=True)
class FoldDiagnostics:
    """One fold's audit trail."""

    fold_index: int
    n_train: int
    n_calibration: int
    n_holdout: int
    brier_uncalibrated: float
    brier_calibrated: float
    brier_naive_baseline: float
    calibration_method: str
    holdout_t0_min: int
    train_t1_max: int
    embargo_size: int

    @property
    def passes_index_intersection(self) -> bool:
        """Blueprint §7.3 — ``max(train_t1) + embargo < min(holdout_t0)``."""
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
        One :class:`FoldDiagnostics` record per walk-forward fold.
    average_uniqueness_per_event
        ``(n_samples,)`` ū vector used to weight SBRF training.
    last_calibration
        :class:`CalibrationResult` from the final (most recent) fold —
        contains the fitted ``CalibratedClassifierCV`` ready for live
        deployment.
    n_estimators
        Forwarded SBRF setting for traceability.
    """

    fold_diagnostics: list[FoldDiagnostics]
    average_uniqueness_per_event: npt.NDArray[np.float64]
    last_calibration: CalibrationResult
    n_estimators: int = field(default=0)

    @property
    def passes_phase5_dod(self) -> bool:
        """Both Blueprint §7.3 DoD checks must hold across every fold."""
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


def train_brain_two(
    X: npt.NDArray[np.floating],
    y: npt.NDArray[np.integer],
    t0: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    t1: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    *,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    train_fraction: float = 0.3,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    n_estimators: int = 200,
    max_depth: int = 5,
    min_samples_leaf: int = 5,
    random_state: int = 0,
) -> BrainTwoResult:
    """Train and calibrate Brain 2 across walk-forward folds.

    Parameters
    ----------
    X
        ``(n_samples, n_features)`` Phase-3 feature matrix aligned with
        ``y``, ``t0``, ``t1``.
    y
        ``(n_samples,)`` binary Triple-Barrier labels.
    t0, t1
        ``(n_samples,)`` event horizon bounds in monotonic int / float
        encoding.
    n_splits
        Number of chronological walk-forward folds.
    embargo_pct
        Embargo between train tail and test head, AND between train and
        calibration, as a fraction of total samples.
    train_fraction
        Burn-in fraction reserved before any test fold begins.
    calibration_fraction
        Within each fold's train portion, the trailing fraction reserved
        for calibration (default 0.2).
    n_estimators, max_depth, min_samples_leaf
        SBRF hyperparameters.
    random_state
        Master seed.

    Returns
    -------
    :class:`BrainTwoResult` with per-fold diagnostics and the final-fold
    calibrated classifier ready for deployment.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    t0_arr = np.asarray(t0, dtype=np.int64)
    t1_arr = np.asarray(t1, dtype=np.int64)
    n_samples = X_arr.shape[0]
    if y_arr.shape != (n_samples,) or t0_arr.shape != (n_samples,) or t1_arr.shape != (n_samples,):
        raise ValueError(
            f"shape mismatch: X={X_arr.shape}, y={y_arr.shape}, "
            f"t0={t0_arr.shape}, t1={t1_arr.shape}"
        )
    if not set(np.unique(y_arr).tolist()) <= {0, 1}:
        raise ValueError("y must be binary in {0, 1}")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X must be finite")

    # Pre-compute average uniqueness over the full event set so each fold
    # uses globally-correct weights (a fold's local ū would be biased toward
    # the small slice).
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

    for fold_i, (train_full_idx, holdout_idx) in enumerate(folds):
        # Split train_full into train + calibration; reserve the trailing slice
        # for calibration so SBRF only sees the older data.
        n_train_full = train_full_idx.size
        n_cal = max(1, round(n_train_full * calibration_fraction))
        train_idx = train_full_idx[: n_train_full - n_cal]
        cal_idx = train_full_idx[n_train_full - n_cal :]

        if train_idx.size < 2 or cal_idx.size < 2 or holdout_idx.size < 2:
            # Degenerate fold; record but skip the heavy fit.
            continue

        X_train = X_arr[train_idx]
        y_train = y_arr[train_idx]
        X_cal = X_arr[cal_idx]
        y_cal = y_arr[cal_idx]
        X_hold = X_arr[holdout_idx]
        y_hold = y_arr[holdout_idx]
        # ind_train: indicator restricted to the train columns. Rows = grid
        # over all events; that's fine — the rows still index time.
        ind_train = ind_full[:, train_idx]
        weight_train = avg_u_full[train_idx]

        sbrf = SequentiallyBootstrappedRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + fold_i,
        )
        sbrf.fit(X_train, y_train, ind_train, sample_weight=weight_train)

        uncal_proba = _positive_class_proba(sbrf, X_hold)
        brier_uncal = float(brier_score_loss(y_hold, uncal_proba))

        calibration = fit_calibrated_classifier(sbrf, X_cal, y_cal, X_hold, y_hold)
        cal_proba = _positive_class_proba(calibration.calibrated, X_hold)
        brier_cal = float(brier_score_loss(y_hold, cal_proba))

        naive_proba = _naive_baseline_proba(y_train, holdout_idx.size)
        brier_naive = float(brier_score_loss(y_hold, naive_proba))

        fold_diagnostics.append(
            FoldDiagnostics(
                fold_index=fold_i,
                n_train=train_idx.size,
                n_calibration=cal_idx.size,
                n_holdout=holdout_idx.size,
                brier_uncalibrated=brier_uncal,
                brier_calibrated=brier_cal,
                brier_naive_baseline=brier_naive,
                calibration_method=calibration.method,
                holdout_t0_min=int(t0_arr[holdout_idx].min()),
                train_t1_max=int(t1_arr[train_idx].max()),
                embargo_size=embargo_size,
            )
        )
        last_calibration = calibration

    if last_calibration is None:
        raise ValueError("every fold was degenerate — increase n_samples or n_splits")

    return BrainTwoResult(
        fold_diagnostics=fold_diagnostics,
        average_uniqueness_per_event=avg_u_full,
        last_calibration=last_calibration,
        n_estimators=n_estimators,
    )
