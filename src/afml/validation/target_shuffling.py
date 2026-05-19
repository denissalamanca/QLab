"""Target shuffling leakage gate (Blueprint §8.3).

If a model retains predictive power on **randomised** labels, the only
plausible explanation is leakage somewhere in the pipeline — the features
must be encoding the labels (or labels' temporal neighbours) directly.
This module exposes :func:`target_shuffling_test` which repeatedly retrains
on permuted labels and confirms the resulting OOS performance is
statistically indistinguishable from chance.

Algorithm:

1. Fit the candidate estimator on ``(X, y, t0, t1)`` via Purged K-Fold;
   compute the **real** OOS Brier score ``B_real``.
2. For ``n_shuffles`` repetitions: permute ``y`` (preserving the
   ``t0 / t1`` order), refit, compute OOS Brier ``B_shuffled[i]``.
3. Empirical p-value: ``p = mean(B_shuffled ≤ B_real)``. Under the null
   "no real signal", ``B_real`` should not be reliably lower than the
   shuffled Briers, so ``p`` should be near ``0.5``.
4. ``p < alpha`` ⇒ the model truly carries signal. ``p ≥ alpha`` AND
   ``B_real`` is competitive with the shuffled distribution ⇒ leakage
   suspected → ``DataLeakageError``.

The leakage test fires when the **shuffled** model performs as well as the
real model — meaning the labels' identity doesn't matter, only their
neighbourhood / leakage does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import brier_score_loss

from afml.selection.purged_kfold import PurgedKFold

DEFAULT_N_SHUFFLES: int = 30
DEFAULT_ALPHA: float = 0.05
PROBA_EPSILON: float = 1e-6
MIN_CLASS_COUNT: int = 2


class DataLeakageError(Exception):
    """Raised by :func:`target_shuffling_test` on a positive leakage finding.

    The strategy producing this error must be **permanently rejected** —
    its OOS signal is indistinguishable from one trained on noise, which
    means the features have memorised the labels' temporal structure
    (path-dependency leakage, future-information leakage, etc.).
    """


@dataclass(frozen=True, slots=True)
class TargetShufflingResult:
    """Diagnostic record from :func:`target_shuffling_test`.

    Attributes
    ----------
    brier_real
        OOS Brier on the true labels.
    brier_shuffled
        Array of OOS Briers from the shuffled refits.
    pvalue
        Empirical p-value: fraction of shuffled Briers ≤ ``brier_real``.
        Under the null, this is uniform on ``[0, 1]``; under a leakage-
        free model it is small.
    n_shuffles
        Number of permutations actually run (degenerate folds may force
        fewer than requested).
    """

    brier_real: float
    brier_shuffled: npt.NDArray[np.float64]
    pvalue: float
    n_shuffles: int

    @property
    def shuffled_mean(self) -> float:
        return float(np.mean(self.brier_shuffled)) if self.brier_shuffled.size else float("nan")

    @property
    def shuffled_std(self) -> float:
        return (
            float(np.std(self.brier_shuffled, ddof=1))
            if self.brier_shuffled.size > 1
            else float("nan")
        )


def _positive_class_proba(classifier: Any, X: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    proba = classifier.predict_proba(X)
    classes = np.asarray(classifier.classes_)
    if proba.shape[1] == 1:
        only_class = int(classes[0])
        return np.full(X.shape[0], 1.0 if only_class == 1 else 0.0, dtype=np.float64)
    pos_col = int(np.where(classes == 1)[0][0])
    return np.asarray(proba[:, pos_col], dtype=np.float64)


def _purged_oof_brier(
    estimator_factory: Callable[[], Any],
    X: npt.NDArray[np.floating],
    y: npt.NDArray[np.integer],
    t0: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    t1: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    *,
    n_splits: int,
    embargo_pct: float,
) -> float:
    """Mean out-of-fold Brier across all PurgedKFold folds."""
    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    fold_briers: list[float] = []
    for train_idx, test_idx in cv.split(t0, t1):
        if train_idx.size < MIN_CLASS_COUNT or test_idx.size < MIN_CLASS_COUNT:
            continue
        train_classes = np.bincount(y[train_idx], minlength=2)
        test_classes = np.bincount(y[test_idx], minlength=2)
        if train_classes.min() < MIN_CLASS_COUNT or test_classes.min() < MIN_CLASS_COUNT:
            continue
        est = estimator_factory()
        est.fit(X[train_idx], y[train_idx])
        proba = _positive_class_proba(est, X[test_idx])
        proba = np.clip(proba, PROBA_EPSILON, 1.0 - PROBA_EPSILON)
        fold_briers.append(float(brier_score_loss(y[test_idx], proba)))
    if not fold_briers:
        return float("nan")
    return float(np.mean(fold_briers))


def target_shuffling_test(
    estimator_factory: Callable[[], Any],
    X: npt.NDArray[np.floating],
    y: npt.NDArray[np.integer],
    t0: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    t1: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    *,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    alpha: float = DEFAULT_ALPHA,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    random_state: int = 0,
    raise_on_leakage: bool = True,
) -> TargetShufflingResult:
    """Run the target-shuffling leakage gate.

    Parameters
    ----------
    estimator_factory
        Zero-argument callable returning a fresh sklearn-style classifier
        with ``fit(X, y)`` and ``predict_proba(X)``. The factory must
        produce a new instance on every call (the shuffles refit
        independently).
    X, y, t0, t1
        Standard Phase 6 inputs.
    n_shuffles
        Number of label-permutation rounds.
    alpha
        Significance level for the leakage decision. If the real Brier is
        not meaningfully lower than the shuffled mean (``p ≥ alpha``), and
        the real Brier is competitive (≤ shuffled mean), we conclude
        leakage.
    n_splits, embargo_pct
        PurgedKFold knobs for the OOF Brier computation.
    random_state
        Master seed; each shuffle uses ``random_state + i``.
    raise_on_leakage
        If True (default), raise :class:`DataLeakageError` on a positive
        finding. If False, just return the result with the diagnostic.

    Returns
    -------
    :class:`TargetShufflingResult`.

    Raises
    ------
    DataLeakageError
        When ``raise_on_leakage`` is True and the test concludes leakage
        (``brier_real >= shuffled_mean`` so the model carries no real
        signal — yet earlier validation showed it predicted well, the
        contradiction implies leakage).
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    t0_arr = np.asarray(t0)
    t1_arr = np.asarray(t1)
    if not set(np.unique(y_arr).tolist()) <= {0, 1}:
        raise ValueError("y must be binary in {0, 1}")
    if n_shuffles < 1:
        raise ValueError(f"n_shuffles must be ≥ 1, got {n_shuffles}")

    brier_real = _purged_oof_brier(
        estimator_factory,
        X_arr,
        y_arr,
        t0_arr,
        t1_arr,
        n_splits=n_splits,
        embargo_pct=embargo_pct,
    )

    shuffled_briers: list[float] = []
    master_rng = np.random.default_rng(random_state)
    for i in range(n_shuffles):
        shuffle_rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1) ^ i)
        y_shuffled = shuffle_rng.permutation(y_arr)
        brier_i = _purged_oof_brier(
            estimator_factory,
            X_arr,
            y_shuffled,
            t0_arr,
            t1_arr,
            n_splits=n_splits,
            embargo_pct=embargo_pct,
        )
        if np.isfinite(brier_i):
            shuffled_briers.append(brier_i)

    shuffled_arr = np.asarray(shuffled_briers, dtype=np.float64)
    if shuffled_arr.size == 0:
        raise ValueError("every shuffled fit failed — check n_splits / class balance")

    # Lower Brier ⇒ better calibration. Real model is "predictively useful"
    # iff its Brier is in the LEFT tail of the shuffled distribution.
    pvalue = float(np.mean(shuffled_arr <= brier_real))

    if raise_on_leakage and pvalue >= alpha:
        raise DataLeakageError(
            f"target shuffling p-value {pvalue:.4f} ≥ alpha={alpha}: "
            f"real Brier {brier_real:.4f} is statistically indistinguishable from "
            f"the shuffled-label Brier distribution (mean={shuffled_arr.mean():.4f}, "
            f"std={shuffled_arr.std(ddof=1):.4f}). The model carries no real signal — "
            f"any earlier OOS performance must have come from leakage. Strategy "
            f"permanently rejected per Blueprint §8.3."
        )

    return TargetShufflingResult(
        brier_real=brier_real,
        brier_shuffled=shuffled_arr,
        pvalue=pvalue,
        n_shuffles=shuffled_arr.size,
    )
