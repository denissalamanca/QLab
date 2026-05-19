"""Cohort construction for PBO — query the Alpha Registry for related trials.

AFML Phase 0-6 audit V1: the Probability of Backtest Overfitting is a
*relative* statistic. It asks "does the in-sample-best strategy keep its
edge out-of-sample, relative to its peers?" A single strategy in a vacuum
has no peers — the PBO degenerates. The fix is to evaluate PBO over a
**cohort**: all trials sharing an ``(asset, algorithmic_family)`` lineage in
the Alpha Registry.

This module bridges the registry to the CPCV performance matrices:

- :func:`count_cohort_trials` — how many trials the cohort holds. Drives both
  the PBO matrix width and the DSR multiple-testing denominator.
- :func:`build_cohort_performance_matrices` — run each cohort strategy
  through CPCV and assemble the ``(n_splits × n_strategies)`` in-sample /
  out-of-sample performance matrices that :func:`afml.validation.compute_pbo`
  consumes.

The registry stores only scalar summaries per experiment, not the trained
estimators, so the actual per-fold performance must be (re)computed from the
candidate factories the caller still holds in memory. The registry's role is
to provide the authoritative *trial count* (the multiple-testing denominator)
and to confirm the cohort is genuinely multi-strategy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import brier_score_loss

from afml.core.registry import AlphaRegistryRepository
from afml.validation.cpcv import CombinatoriallyPurgedKFold
from afml.validation.pbo import MIN_COHORT_STRATEGIES

PROBA_EPSILON: float = 1e-6
MIN_CLASS_COUNT: int = 2


def count_cohort_trials(
    registry: AlphaRegistryRepository,
    *,
    asset: str | None = None,
    algorithmic_family: str | None = None,
) -> int:
    """Return the number of registered trials in a cohort.

    Thin, intention-revealing wrapper over
    :meth:`AlphaRegistryRepository.trials_for`. With both filters ``None``
    this returns the lab-wide total — the canonical DSR multiple-testing
    denominator (every hypothesis ever tried, including ``FAILED_AT_MDA``).
    Narrowing by ``algorithmic_family`` gives the PBO peer-cohort size.
    """
    return registry.trials_for(asset=asset, family=algorithmic_family)


def _positive_class_proba(classifier: Any, X: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    proba = classifier.predict_proba(X)
    classes = np.asarray(classifier.classes_)
    if proba.shape[1] == 1:
        only_class = int(classes[0])
        return np.full(X.shape[0], 1.0 if only_class == 1 else 0.0, dtype=np.float64)
    pos_col = int(np.where(classes == 1)[0][0])
    return np.asarray(proba[:, pos_col], dtype=np.float64)


def build_cohort_performance_matrices(
    cohort_factories: list[Callable[[], Any]],
    X: npt.NDArray[np.floating],
    y: npt.NDArray[np.integer],
    t0: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    t1: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run a cohort through CPCV and assemble PBO performance matrices.

    AFML Phase 0-6 audit V1: produces the ``(n_splits × n_strategies)``
    in-sample / out-of-sample matrices ``compute_pbo`` needs, where columns
    are the cohort strategies and rows are CPCV combinatorial splits.

    Performance is the **negative Brier score** (higher = better) so PBO's
    "argmax in-sample" picks the best-calibrated strategy.

    Parameters
    ----------
    cohort_factories
        ``≥ MIN_COHORT_STRATEGIES`` zero-arg callables, each returning a
        fresh sklearn-style classifier. Represents the family cohort.
    X, y, t0, t1
        Standard inputs; ``t1`` is the realised barrier-touch horizon.
    n_groups, n_test_groups, embargo_pct
        CPCV knobs.

    Returns
    -------
    ``(in_sample_perf, out_sample_perf)`` — both ``(n_valid_splits ×
    n_strategies)`` float64 matrices, ready for :func:`compute_pbo`.

    Raises
    ------
    ValueError
        If the cohort has fewer than ``MIN_COHORT_STRATEGIES`` members or
        every CPCV split was degenerate.
    """
    n_strategies = len(cohort_factories)
    if n_strategies < MIN_COHORT_STRATEGIES:
        raise ValueError(
            f"cohort must hold ≥ {MIN_COHORT_STRATEGIES} strategies for PBO, "
            f"got {n_strategies} (AFML Phase 0-6 audit V1)"
        )
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    t0_arr = np.asarray(t0, dtype=np.int64)
    t1_arr = np.asarray(t1, dtype=np.int64)

    cv = CombinatoriallyPurgedKFold(
        n_groups=n_groups, n_test_groups=n_test_groups, embargo_pct=embargo_pct
    )
    folds = list(cv.split(t0_arr, t1_arr))
    n_splits = len(folds)
    is_perf = np.full((n_splits, n_strategies), np.nan, dtype=np.float64)
    oos_perf = np.full((n_splits, n_strategies), np.nan, dtype=np.float64)

    for fold_i, fold in enumerate(folds):
        if fold.train_idx.size < MIN_CLASS_COUNT or fold.test_idx.size < MIN_CLASS_COUNT:
            continue
        train_classes = np.bincount(y_arr[fold.train_idx], minlength=2)
        test_classes = np.bincount(y_arr[fold.test_idx], minlength=2)
        if train_classes.min() < MIN_CLASS_COUNT or test_classes.min() < MIN_CLASS_COUNT:
            continue
        X_train, X_test = X_arr[fold.train_idx], X_arr[fold.test_idx]
        y_train, y_test = y_arr[fold.train_idx], y_arr[fold.test_idx]
        for s_idx, factory in enumerate(cohort_factories):
            est = factory()
            try:
                est.fit(X_train, y_train)
                p_train = np.clip(
                    _positive_class_proba(est, X_train), PROBA_EPSILON, 1.0 - PROBA_EPSILON
                )
                p_test = np.clip(
                    _positive_class_proba(est, X_test), PROBA_EPSILON, 1.0 - PROBA_EPSILON
                )
            except (ValueError, RuntimeError):
                continue
            # Negative Brier — higher is better, so argmax = best calibrated.
            is_perf[fold_i, s_idx] = -float(brier_score_loss(y_train, p_train))
            oos_perf[fold_i, s_idx] = -float(brier_score_loss(y_test, p_test))

    valid_rows = ~(np.isnan(is_perf).any(axis=1) | np.isnan(oos_perf).any(axis=1))
    if not valid_rows.any():
        raise ValueError("every CPCV split was degenerate — check n_groups / class balance")
    return is_perf[valid_rows], oos_perf[valid_rows]
