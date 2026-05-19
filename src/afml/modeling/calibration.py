"""Brier-minimizing probability calibration.

Blueprint §7.2: ``CalibratedClassifierCV`` with both ``isotonic`` and
``sigmoid`` (Platt) — keep the one that minimises the Brier score on an
independent holdout.

Why ``FrozenEstimator``? sklearn's default ``CalibratedClassifierCV`` uses an
internal ``KFold`` which violates the AFML purging / embargo requirement
(CLAUDE.md banned-methods table). The supported escape hatch in sklearn ≥ 1.6
is to wrap the prefit base estimator in :class:`sklearn.frozen.FrozenEstimator`
— this tells ``CalibratedClassifierCV`` not to re-fit and to use the entire
calibration set as the calibration sample directly. Three disjoint time slices
per fold:

::

    [-------- train --------][embargo][--- calibrate ---][embargo][--- holdout ---]

The pipeline orchestrator (:mod:`afml.modeling.pipeline`) materialises these
slices from the chronological event index using
:class:`afml.selection.PurgedWalkForwardCV`.

This module exposes:

- :func:`fit_calibrated_classifier` — fit isotonic + sigmoid on a held-out
  calibration set and return the winner with diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

CalibrationMethod = str  # Literal["isotonic", "sigmoid"]
PROBA_EPSILON: float = 1e-6


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
        The fitted ``CalibratedClassifierCV`` corresponding to the winner.
    """

    method: CalibrationMethod
    brier_isotonic: float
    brier_sigmoid: float
    brier_uncalibrated: float
    calibrated: CalibratedClassifierCV


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


def fit_calibrated_classifier(
    base_estimator: Any,
    X_calibration: npt.NDArray[np.floating],
    y_calibration: npt.NDArray[np.integer],
    X_holdout: npt.NDArray[np.floating],
    y_holdout: npt.NDArray[np.integer],
) -> CalibrationResult:
    """Fit isotonic + sigmoid on the calibration set, score on the holdout.

    Parameters
    ----------
    base_estimator
        A **prefit** classifier exposing ``predict_proba``. The estimator
        must already have been trained on a non-overlapping training set.
    X_calibration, y_calibration
        Held-out calibration set (post-embargo from training).
    X_holdout, y_holdout
        Independent holdout (post-embargo from calibration). Used **only**
        to score the two calibration variants.

    Returns
    -------
    :class:`CalibrationResult` carrying the winning method, the three Brier
    scores, and the fitted ``CalibratedClassifierCV``.

    Raises
    ------
    ValueError
        If shapes mismatch or ``y_*`` are not binary.
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

    # Uncalibrated baseline.
    uncal_proba = _positive_class_proba(base_estimator, X_hold)
    brier_uncalibrated = float(brier_score_loss(y_hold, uncal_proba))

    # Wrap the prefit estimator so CalibratedClassifierCV won't re-fit it
    # (sklearn ≥ 1.6 retired `cv="prefit"` in favour of FrozenEstimator).
    # This is the supported, KFold-free path.
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
        winner = calibrated_iso
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
