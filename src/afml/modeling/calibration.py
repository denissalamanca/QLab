"""Brier-minimising probability calibration with **purged** cross-fitting.

Blueprint §7.2 + AFML 0-5 integration audit V1.

The naive use of :class:`sklearn.calibration.CalibratedClassifierCV` defaults
to a stratified KFold for cross-fitting the calibrator, which **randomly
shuffles** non-IID financial labels. That is a silent leakage trap — the
calibrator sees future-overlapping data, producing artificially low Brier
scores and unrealistic confidence at deployment. AFML's prescription is to
pass an explicit purged + embargoed splitter into the ``cv`` argument so the
calibrator is fit on chronologically-clean out-of-fold predictions.

This module exposes two complementary APIs:

1. :func:`fit_calibrated_classifier_with_purged_cv` — the canonical entry
   point. For SBRF (whose ``fit`` requires the ``indicator_mat`` positional
   argument that sklearn's CV machinery cannot pass through), it runs a
   **manual** purged cross-fitting loop that explicitly invokes
   ``PurgedKFold.split(t0, t1)``. For estimators with a standard
   ``fit(X, y, sample_weight=...)`` signature it delegates to
   :class:`CalibratedClassifierCV` with ``cv=PurgedKFoldSklearn(t0, t1)``.

2. :func:`fit_calibrated_classifier` — backward-compatible helper that uses a
   pre-fit base estimator + held-out calibration set via
   :class:`sklearn.frozen.FrozenEstimator`. Retained because some callers
   want the simpler hold-out flow (no internal CV at all is also leakage-safe
   so long as ``X_calibration`` is post-embargo from training).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from afml.selection.purged_kfold import PurgedKFold, PurgedKFoldSklearn

CalibrationMethod = str  # Literal["isotonic", "sigmoid"]
PROBA_EPSILON: float = 1e-6
# Calibrator regularisation — Platt scaling traditionally uses C=1e10
# (effectively unregularised); isotonic has no hyperparam to set here.
PLATT_C: float = 1e10
MIN_CLASS_COUNT: int = 2


def _normalize_sample_weights(weights: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Scale uniqueness weights so they sum to ``N`` (number of samples).

    AFML 0-5 audit pre-Phase-6 patch: raw average-uniqueness weights
    ``ū_i ∈ (0, 1]`` are fractions. Passed directly to XGBoost they can
    trigger vanishing gradients via ``min_child_weight`` (default 1.0 —
    a leaf with weight-sum < 1 is rejected, so a uniqueness-weighted leaf
    needs ``Σ ū ≥ 1`` which is roughly ``2/mean(ū)`` samples instead of 1).

    The fix is to scale by ``N / Σ ū`` so the *aggregate* weight matches
    an unweighted fit while the per-sample *ratio* of weights is preserved.
    sklearn tree-based estimators are invariant to this overall scale, but
    XGBoost's ``min_child_weight`` is not — normalisation is therefore
    mandatory at the boundary, not just nice-to-have.

    Returns a fresh ``float64`` array. Degenerate input (all zeros) is
    handled by returning uniform ones.
    """
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    if total <= 0.0:
        return np.ones(w.shape, dtype=np.float64)
    return w * (float(w.size) / total)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Diagnostic record of the isotonic-vs-sigmoid contest.

    Attributes
    ----------
    method
        Winning calibration method (``"isotonic"`` or ``"sigmoid"``).
    brier_isotonic, brier_sigmoid
        Brier scores on the holdout set.
    brier_uncalibrated
        Brier score of the raw base classifier on the same holdout, for
        comparison.
    calibrated
        The fitted calibrator. Exposes a ``predict_proba(X)`` method
        returning ``(n_samples, 2)`` probability columns ordered as
        ``[P(y=0), P(y=1)]``.
    """

    method: CalibrationMethod
    brier_isotonic: float
    brier_sigmoid: float
    brier_uncalibrated: float
    calibrated: Any


class SBRFLikeEstimator(Protocol):
    """Anything with the SBRF triple-argument fit signature."""

    def fit(
        self,
        X: npt.NDArray[np.floating],
        y: npt.NDArray[np.integer],
        indicator_mat: npt.NDArray[np.integer],
        *,
        sample_weight: npt.NDArray[np.floating] | None = None,
    ) -> Any: ...

    def predict_proba(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]: ...


def _positive_class_proba(
    classifier: Any,
    X: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """Extract ``P(y=1)`` robustly even for single-class fits."""
    proba = classifier.predict_proba(X)
    classes = np.asarray(classifier.classes_)
    if proba.shape[1] == 1:
        only_class = int(classes[0])
        return np.full(X.shape[0], 1.0 if only_class == 1 else 0.0, dtype=np.float64)
    pos_col = int(np.where(classes == 1)[0][0])
    return np.asarray(proba[:, pos_col], dtype=np.float64)


@dataclass
class PurgedCVCalibratedSBRF:
    """A trained SBRF plus a separately-fit calibrator over OOF probabilities.

    Mimics the ``predict_proba`` surface of :class:`CalibratedClassifierCV` so
    the rest of the Brain 2 pipeline (and Phase 7 bet-sizer) treat it as a
    plain probabilistic classifier.
    """

    base_estimator: Any
    calibrator: Any  # IsotonicRegression or LogisticRegression
    method: CalibrationMethod
    classes_: npt.NDArray[np.int64]

    def predict_proba(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        raw = _positive_class_proba(self.base_estimator, X)
        if self.method == "isotonic":
            cal_pos = self.calibrator.predict(raw)
        else:
            cal_pos = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        cal_pos = np.clip(cal_pos, PROBA_EPSILON, 1.0 - PROBA_EPSILON)
        out = np.column_stack([1.0 - cal_pos, cal_pos])
        return np.asarray(out, dtype=np.float64)

    def predict(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.int64]:
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=np.int64)


def _fit_calibrator(
    method: CalibrationMethod,
    raw_proba: npt.NDArray[np.float64],
    y: npt.NDArray[np.integer],
) -> Any:
    """Fit a 1-D calibrator on ``(raw_proba, y)``.

    - ``isotonic``: :class:`sklearn.isotonic.IsotonicRegression` (monotone,
      non-parametric).
    - ``sigmoid``: :class:`sklearn.linear_model.LogisticRegression` (Platt
      scaling — a 2-parameter logistic on the raw probability).
    """
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(raw_proba, y)
        return cal
    if method == "sigmoid":
        cal = LogisticRegression(C=PLATT_C, max_iter=1000)
        cal.fit(raw_proba.reshape(-1, 1), y)
        return cal
    raise ValueError(f"unknown calibration method {method!r}")


def fit_calibrated_sbrf_with_purged_cv(
    base_estimator: SBRFLikeEstimator,
    X_train: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.integer],
    indicator_mat_train: npt.NDArray[np.integer],
    t0_train: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    t1_train: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    sample_weight_train: npt.NDArray[np.floating],
    X_holdout: npt.NDArray[np.floating],
    y_holdout: npt.NDArray[np.integer],
    *,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> CalibrationResult:
    """Fit isotonic + sigmoid calibrators via **purged** cross-fitting.

    AFML 0-5 audit V1: explicitly invokes :meth:`PurgedKFold.split` to generate
    train / out-of-fold pairs; the base SBRF is refit on each train fold (so
    its sequential bootstrap + sample-weight propagation stay intact); the
    calibrator (1-D isotonic or Platt) is fit on the concatenated OOF
    probabilities.

    The audit test
    ``test_calibration_uses_purged_kfold_split`` confirms that
    ``PurgedKFold.split`` is the entry point of the cross-fitting loop.

    Parameters
    ----------
    base_estimator
        Any estimator with SBRF's three-argument fit signature
        ``fit(X, y, indicator_mat, sample_weight=...)``.
        The instance is used as a *template* — :func:`sklearn.base.clone`-style
        re-instantiation is not needed because the SBRF is stateless until
        fit.
    X_train, y_train
        Training feature matrix and binary labels.
    indicator_mat_train
        Phase-2 indicator matrix restricted to the train columns.
    t0_train, t1_train
        Per-sample horizons for the purged splitter (Phase 2 ``exit_timestamp``
        is the canonical t1 — Phase 0-4 audit V1).
    sample_weight_train
        Average-uniqueness weights ū_i for the train rows. Passed to the SBRF
        on every fold (AFML 0-5 audit V3).
    X_holdout, y_holdout
        Independent post-embargo holdout used to score the two calibration
        variants and pick the Brier-min winner.
    n_splits, embargo_pct
        Purged K-Fold knobs.

    Returns
    -------
    :class:`CalibrationResult` with the winning calibrator wrapped in
    :class:`PurgedCVCalibratedSBRF`.
    """
    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.int64)
    ind_tr = np.asarray(indicator_mat_train, dtype=np.int64)
    sw_tr = np.asarray(sample_weight_train, dtype=np.float64)
    X_hold = np.asarray(X_holdout, dtype=np.float64)
    y_hold = np.asarray(y_holdout, dtype=np.int64)

    n_train = X_tr.shape[0]
    if y_tr.shape != (n_train,) or sw_tr.shape != (n_train,):
        raise ValueError("y_train and sample_weight_train must match X_train rows")
    if ind_tr.shape[1] != n_train:
        raise ValueError(f"indicator_mat_train columns {ind_tr.shape[1]} != X_train rows {n_train}")

    # AFML 0-5 audit pre-Phase-6 patch: normalise ū_i so the aggregate
    # weight equals N. Preserves the relative weighting structure while
    # preventing XGBoost ``min_child_weight`` from suppressing fractional-
    # weight leaves. The slicing below uses the normalised vector.
    sw_tr = _normalize_sample_weights(sw_tr)

    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    oof_proba = np.full(n_train, np.nan, dtype=np.float64)

    # Cross-fitted out-of-fold probabilities — sklearn's CalibratedClassifierCV
    # builds these too, but with a stratified KFold we can't override (SBRF's
    # indicator_mat is not accepted by its fit_params). Manual loop:
    fold_count = 0
    for train_idx, oof_idx in cv.split(t0_train, t1_train):
        fold_count += 1
        if train_idx.size < MIN_CLASS_COUNT or oof_idx.size == 0:
            continue
        # Per-fold class diversity check.
        train_classes = np.bincount(y_tr[train_idx], minlength=2)
        if train_classes.min() < MIN_CLASS_COUNT:
            continue

        sbrf_fold = _clone_sbrf(base_estimator, fold_seed_offset=fold_count)
        sbrf_fold.fit(
            X_tr[train_idx],
            y_tr[train_idx],
            ind_tr[:, train_idx],
            sample_weight=sw_tr[train_idx],
        )
        oof_proba[oof_idx] = _positive_class_proba(sbrf_fold, X_tr[oof_idx])

    valid_mask = ~np.isnan(oof_proba)
    if valid_mask.sum() < MIN_CLASS_COUNT or len(np.unique(y_tr[valid_mask])) < MIN_CLASS_COUNT:
        raise ValueError(
            "purged cross-fitting produced too few OOF probabilities or only one class; "
            "increase n_splits, decrease embargo, or check class balance"
        )

    # Fit base estimator once more on the FULL training set — this is the
    # production model the calibrator wraps. Sample weights still applied.
    base_full = _clone_sbrf(base_estimator, fold_seed_offset=0)
    base_full.fit(X_tr, y_tr, ind_tr, sample_weight=sw_tr)

    # Uncalibrated baseline on the holdout.
    raw_holdout = _positive_class_proba(base_full, X_hold)
    brier_uncalibrated = float(brier_score_loss(y_hold, raw_holdout))

    # Fit isotonic + sigmoid calibrators on the OOF predictions.
    cal_iso = _fit_calibrator("isotonic", oof_proba[valid_mask], y_tr[valid_mask])
    cal_sig = _fit_calibrator("sigmoid", oof_proba[valid_mask], y_tr[valid_mask])

    wrapper_iso = PurgedCVCalibratedSBRF(
        base_estimator=base_full,
        calibrator=cal_iso,
        method="isotonic",
        classes_=np.asarray([0, 1], dtype=np.int64),
    )
    wrapper_sig = PurgedCVCalibratedSBRF(
        base_estimator=base_full,
        calibrator=cal_sig,
        method="sigmoid",
        classes_=np.asarray([0, 1], dtype=np.int64),
    )

    iso_proba = _positive_class_proba(wrapper_iso, X_hold)
    sig_proba = _positive_class_proba(wrapper_sig, X_hold)
    brier_iso = float(brier_score_loss(y_hold, iso_proba))
    brier_sig = float(brier_score_loss(y_hold, sig_proba))

    if brier_iso <= brier_sig:
        winner: PurgedCVCalibratedSBRF = wrapper_iso
        method: CalibrationMethod = "isotonic"
    else:
        winner = wrapper_sig
        method = "sigmoid"

    return CalibrationResult(
        method=method,
        brier_isotonic=brier_iso,
        brier_sigmoid=brier_sig,
        brier_uncalibrated=brier_uncalibrated,
        calibrated=winner,
    )


def fit_calibrated_classifier_with_purged_cv(
    base_estimator: Any,
    X_train: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.integer],
    t0_train: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    t1_train: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    sample_weight_train: npt.NDArray[np.floating],
    X_holdout: npt.NDArray[np.floating],
    y_holdout: npt.NDArray[np.integer],
    *,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> CalibrationResult:
    """Calibrate any standard sklearn classifier via PurgedKFold cross-fitting.

    Uses :class:`sklearn.calibration.CalibratedClassifierCV` with
    ``cv=PurgedKFoldSklearn(t0_train, t1_train, ...)``. Sample weights
    propagate through ``fit_params``.

    This is the **XGBoost / vanilla RandomForest** path. For SBRF, use
    :func:`fit_calibrated_sbrf_with_purged_cv` instead (SBRF's
    ``indicator_mat`` argument cannot be threaded through sklearn's CV
    machinery).
    """
    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.int64)
    sw_tr = np.asarray(sample_weight_train, dtype=np.float64)
    X_hold = np.asarray(X_holdout, dtype=np.float64)
    y_hold = np.asarray(y_holdout, dtype=np.int64)

    n_train = X_tr.shape[0]
    if y_tr.shape != (n_train,) or sw_tr.shape != (n_train,):
        raise ValueError("y_train and sample_weight_train must match X_train rows")

    # AFML 0-5 audit pre-Phase-6 patch: normalise sample weights to sum to
    # n_train so XGBoost's ``min_child_weight`` doesn't squash leaves.
    sw_tr = _normalize_sample_weights(sw_tr)

    cv = PurgedKFoldSklearn(t0_train, t1_train, n_splits=n_splits, embargo_pct=embargo_pct)

    calibrated_iso = CalibratedClassifierCV(base_estimator, method="isotonic", cv=cv)
    calibrated_iso.fit(X_tr, y_tr, sample_weight=sw_tr)
    iso_proba = _positive_class_proba(calibrated_iso, X_hold)
    brier_iso = float(brier_score_loss(y_hold, iso_proba))

    calibrated_sig = CalibratedClassifierCV(base_estimator, method="sigmoid", cv=cv)
    calibrated_sig.fit(X_tr, y_tr, sample_weight=sw_tr)
    sig_proba = _positive_class_proba(calibrated_sig, X_hold)
    brier_sig = float(brier_score_loss(y_hold, sig_proba))

    # Uncalibrated baseline: refit base estimator on full train + score on holdout.
    base_clone = _clone_generic(base_estimator)
    base_clone.fit(X_tr, y_tr, sample_weight=sw_tr)
    uncal = _positive_class_proba(base_clone, X_hold)
    brier_uncalibrated = float(brier_score_loss(y_hold, uncal))

    if brier_iso <= brier_sig:
        winner: Any = calibrated_iso
        method: CalibrationMethod = "isotonic"
    else:
        winner = calibrated_sig
        method = "sigmoid"

    return CalibrationResult(
        method=method,
        brier_isotonic=brier_iso,
        brier_sigmoid=brier_sig,
        brier_uncalibrated=brier_uncalibrated,
        calibrated=winner,
    )


def fit_calibrated_classifier(
    base_estimator: Any,
    X_calibration: npt.NDArray[np.floating],
    y_calibration: npt.NDArray[np.integer],
    X_holdout: npt.NDArray[np.floating],
    y_holdout: npt.NDArray[np.integer],
) -> CalibrationResult:
    """Legacy: fit isotonic + sigmoid on a pre-existing held-out calibration set.

    Retained for callers that have already produced a chronologically clean
    ``(X_calibration, y_calibration)`` slice via an outer purged split. Uses
    :class:`sklearn.frozen.FrozenEstimator` to prevent ``CalibratedClassifierCV``
    from re-fitting the base estimator or doing any internal CV — no risk of
    KFold leakage on this path because no CV happens at all.

    The audit's preferred path is
    :func:`fit_calibrated_sbrf_with_purged_cv` / :func:`fit_calibrated_classifier_with_purged_cv`,
    which run an explicit purged cross-fitting loop and use the FULL training
    set for both base fit and OOF calibration (no calibration-set carve-out).
    """
    X_cal = np.asarray(X_calibration, dtype=np.float64)
    y_cal = np.asarray(y_calibration, dtype=np.int64)
    X_hold = np.asarray(X_holdout, dtype=np.float64)
    y_hold = np.asarray(y_holdout, dtype=np.int64)

    if X_cal.ndim != 2 or X_hold.ndim != 2:
        raise ValueError("X_calibration and X_holdout must both be 2-D")
    if y_cal.shape != (X_cal.shape[0],) or y_hold.shape != (X_hold.shape[0],):
        raise ValueError("y shape mismatched against X")
    for name, y_arr in (("y_calibration", y_cal), ("y_holdout", y_hold)):
        if not set(np.unique(y_arr).tolist()) <= {0, 1}:
            raise ValueError(f"{name} must be binary in {{0, 1}}")

    uncal_proba = _positive_class_proba(base_estimator, X_hold)
    brier_uncalibrated = float(brier_score_loss(y_hold, uncal_proba))

    frozen = FrozenEstimator(base_estimator)

    calibrated_iso = CalibratedClassifierCV(frozen, method="isotonic")
    calibrated_iso.fit(X_cal, y_cal)
    iso_proba = _positive_class_proba(calibrated_iso, X_hold)
    brier_iso = float(brier_score_loss(y_hold, iso_proba))

    calibrated_sig = CalibratedClassifierCV(frozen, method="sigmoid")
    calibrated_sig.fit(X_cal, y_cal)
    sig_proba = _positive_class_proba(calibrated_sig, X_hold)
    brier_sig = float(brier_score_loss(y_hold, sig_proba))

    if brier_iso <= brier_sig:
        method: CalibrationMethod = "isotonic"
        winner: Any = calibrated_iso
    else:
        method = "sigmoid"
        winner = calibrated_sig

    return CalibrationResult(
        method=method,
        brier_isotonic=brier_iso,
        brier_sigmoid=brier_sig,
        brier_uncalibrated=brier_uncalibrated,
        calibrated=winner,
    )


def _clone_sbrf(template: SBRFLikeEstimator, *, fold_seed_offset: int) -> Any:
    """Construct a fresh SBRF-like estimator with the template's init params.

    sklearn's :func:`sklearn.base.clone` would do this for a standard sklearn
    estimator, but we want to bump ``random_state`` per fold so the per-tree
    sequential-bootstrap seeds differ across folds.
    """
    cls = type(template)
    init_params = template.get_params(deep=False)  # type: ignore[attr-defined]
    init_params = dict(init_params)
    base_rs = init_params.get("random_state", 0)
    init_params["random_state"] = (base_rs or 0) + fold_seed_offset
    return cls(**init_params)


def _clone_generic(template: Any) -> Any:
    """Best-effort clone for non-SBRF estimators (RandomForest, XGBoost, ...).

    Uses :func:`sklearn.base.clone` when possible; falls back to instantiating
    the class with ``get_params(deep=False)``.
    """
    try:
        return clone(template)
    except (TypeError, ValueError):
        cls = type(template)
        try:
            params = template.get_params(deep=False)
        except AttributeError:
            params = {}
        return cls(**params)
