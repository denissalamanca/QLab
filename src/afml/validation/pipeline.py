"""Phase 6 orchestrator — full validation gate (Blueprint §8).

Public entry: :func:`validate_strategy`. Given a list of candidate
classifier factories (each producing a fresh estimator on every call), a
feature matrix, labels, event horizons and a realised per-event return
series, the orchestrator:

1. Runs :class:`CombinatoriallyPurgedKFold` over the candidate set,
   collecting per-combination in-sample / out-of-sample performance.
2. Computes :class:`PBOResult` from the IS / OOS matrices.
3. Constructs synthetic OOS prediction paths for the **incumbent**
   (best-Brier) candidate and converts each path into a return series →
   per-path Sharpe ratio.
4. Computes :class:`DSRResult` against the Alpha Registry's trial count
   for the multiple-testing deflation.
5. Runs :func:`target_shuffling_test` on the incumbent — raises
   :class:`DataLeakageError` if the shuffled-label model is competitive.

A strategy passes Phase 6 when:

* ``PBO < pbo_threshold`` (default 0.05),
* ``DSR > dsr_threshold`` (default 0.95 ≈ 95% confidence the SR is real),
* the target-shuffling test returns ``pvalue < alpha`` (no leakage).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import brier_score_loss

from afml.validation.cpcv import CombinatoriallyPurgedKFold
from afml.validation.dsr import DSR_MIN_TRIALS, DSRResult, deflated_sharpe_ratio
from afml.validation.pbo import PBOResult, compute_pbo
from afml.validation.target_shuffling import (
    TargetShufflingResult,
    target_shuffling_test,
)

PROBA_EPSILON: float = 1e-6
DEFAULT_PBO_THRESHOLD: float = 0.05
DEFAULT_DSR_THRESHOLD: float = 0.95
DEFAULT_BET_THRESHOLD: float = 0.5
MIN_CLASS_COUNT: int = 2


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Top-level Phase 6 output.

    Attributes
    ----------
    pbo
        :class:`PBOResult` from CPCV.
    dsr
        :class:`DSRResult` of the incumbent strategy.
    target_shuffling
        :class:`TargetShufflingResult` for the incumbent.
    incumbent_idx
        Position of the winning strategy in the candidate list.
    pbo_threshold, dsr_threshold
        Forwarded thresholds used to compute ``passes_phase6_dod``.
    """

    pbo: PBOResult
    dsr: DSRResult
    target_shuffling: TargetShufflingResult
    incumbent_idx: int
    pbo_threshold: float = field(default=DEFAULT_PBO_THRESHOLD)
    dsr_threshold: float = field(default=DEFAULT_DSR_THRESHOLD)

    @property
    def passes_phase6_dod(self) -> bool:
        """Blueprint §8.3 DoD — PBO < 0.05 AND DSR > threshold AND no leakage.

        A DSR cold-start quarantine (AFML Phase 0-6 audit V2) hard-fails the
        gate regardless of the other metrics: an unvalidated trial population
        cannot be trusted.
        """
        if self.dsr.quarantined:
            return False
        return (
            self.pbo.pbo < self.pbo_threshold
            and self.dsr.dsr > self.dsr_threshold
            and self.target_shuffling.pvalue < DEFAULT_PBO_THRESHOLD
        )


def _positive_class_proba(classifier: Any, X: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    proba = classifier.predict_proba(X)
    classes = np.asarray(classifier.classes_)
    if proba.shape[1] == 1:
        only_class = int(classes[0])
        return np.full(X.shape[0], 1.0 if only_class == 1 else 0.0, dtype=np.float64)
    pos_col = int(np.where(classes == 1)[0][0])
    return np.asarray(proba[:, pos_col], dtype=np.float64)


def _fold_is_oos_brier(
    estimator: Any,
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int64],
    X_test: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.int64],
) -> tuple[float, float]:
    """Train one estimator and return (in-sample Brier, out-of-sample Brier)."""
    estimator.fit(X_train, y_train)
    p_train = np.clip(_positive_class_proba(estimator, X_train), PROBA_EPSILON, 1.0 - PROBA_EPSILON)
    p_test = np.clip(_positive_class_proba(estimator, X_test), PROBA_EPSILON, 1.0 - PROBA_EPSILON)
    brier_is = float(brier_score_loss(y_train, p_train))
    brier_oos = float(brier_score_loss(y_test, p_test))
    return brier_is, brier_oos


def _signal_to_returns(
    probabilities: npt.NDArray[np.float64],
    realized_returns: npt.NDArray[np.float64],
    threshold: float = DEFAULT_BET_THRESHOLD,
) -> npt.NDArray[np.float64]:
    """Convert per-event probabilities into a return series.

    The minimal interpretation used for Phase 6: enter the trade when
    ``P(success) > threshold``, take ``realized_return``; skip otherwise.
    Phase 7 will replace this with a proper probability-sized bet.
    """
    mask = probabilities > threshold
    return np.where(mask, realized_returns, 0.0)


def validate_strategy(  # noqa: PLR0915 — single orchestration; clarity wins
    strategy_factories: list[Callable[[], Any]],
    X: npt.NDArray[np.floating],
    y: npt.NDArray[np.integer],
    t0: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    t1: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    realized_returns: npt.NDArray[np.floating],
    *,
    n_trials: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
    pbo_threshold: float = DEFAULT_PBO_THRESHOLD,
    dsr_threshold: float = DEFAULT_DSR_THRESHOLD,
    n_shuffles: int = 30,
    shuffle_alpha: float = DEFAULT_PBO_THRESHOLD,
    bet_threshold: float = DEFAULT_BET_THRESHOLD,
    sharpe_std_of_trials: float | None = None,
    periods_per_year: int = 252,
    dsr_min_trials: int = DSR_MIN_TRIALS,
    random_state: int = 0,
    raise_on_leakage: bool = True,
) -> ValidationResult:
    """End-to-end Phase 6 validation gate.

    Parameters
    ----------
    strategy_factories
        Candidate strategies. Each entry is a zero-arg callable returning a
        fresh sklearn-style classifier (``fit(X, y)`` + ``predict_proba``).
        The IS-best winner across the CPCV folds becomes the **incumbent**.
    X, y, t0, t1
        Standard Phase 5 inputs (``t1`` is the realised barrier-touch
        timestamp per Phase 0-4 audit V1).
    realized_returns
        Per-event realised return (``return_pct`` from Triple-Barrier).
        Used to translate predictions → strategy returns → Sharpe → DSR.
    n_trials
        Multiple-testing denominator from the Alpha Registry (every
        hypothesis ever tried by this lab, including failed ones).
    n_groups, n_test_groups, embargo_pct
        CPCV knobs.
    pbo_threshold, dsr_threshold
        Phase 6 DoD thresholds.
    n_shuffles, shuffle_alpha
        Target shuffling parameters.
    bet_threshold
        Probability cutoff for translating predictions into trades.
    sharpe_std_of_trials
        Cross-sectional std of historical strategy Sharpes (passed to
        DSR). ``None`` ⇒ default ``1.0``.
    periods_per_year
        Annualisation factor for the Sharpe.
    random_state
        Master seed.
    raise_on_leakage
        Whether to raise :class:`DataLeakageError` on a positive
        shuffling-test finding.

    Returns
    -------
    :class:`ValidationResult`.

    Raises
    ------
    DataLeakageError
        From the target-shuffling step when leakage is detected.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    t0_arr = np.asarray(t0, dtype=np.int64)
    t1_arr = np.asarray(t1, dtype=np.int64)
    r_arr = np.asarray(realized_returns, dtype=np.float64)
    n_samples = X_arr.shape[0]
    n_strategies = len(strategy_factories)
    if n_strategies < 2:
        raise ValueError(f"need ≥ 2 candidate strategies for PBO, got {n_strategies}")
    if y_arr.shape != (n_samples,) or t0_arr.shape != (n_samples,):
        raise ValueError("X / y / t0 shape mismatch")
    if r_arr.shape != (n_samples,):
        raise ValueError(f"realized_returns shape {r_arr.shape} != ({n_samples},)")

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
        # Class balance check.
        train_classes = np.bincount(y_arr[fold.train_idx], minlength=2)
        test_classes = np.bincount(y_arr[fold.test_idx], minlength=2)
        if train_classes.min() < MIN_CLASS_COUNT or test_classes.min() < MIN_CLASS_COUNT:
            continue

        X_train, X_test = X_arr[fold.train_idx], X_arr[fold.test_idx]
        y_train, y_test = y_arr[fold.train_idx], y_arr[fold.test_idx]
        for s_idx, factory in enumerate(strategy_factories):
            est = factory()
            try:
                brier_is, brier_oos = _fold_is_oos_brier(est, X_train, y_train, X_test, y_test)
            except (ValueError, RuntimeError):
                continue
            # Lower Brier = better; PBO works on "higher is better" → negate.
            is_perf[fold_i, s_idx] = -brier_is
            oos_perf[fold_i, s_idx] = -brier_oos

    valid_rows = ~(np.isnan(is_perf).any(axis=1) | np.isnan(oos_perf).any(axis=1))
    if not valid_rows.any():
        raise ValueError("every CPCV fold was degenerate — check n_splits / n_groups")
    pbo_result = compute_pbo(is_perf[valid_rows], oos_perf[valid_rows])

    # Incumbent = best mean OOS perf across valid folds.
    mean_oos = np.nanmean(oos_perf[valid_rows], axis=0)
    incumbent_idx = int(np.argmax(mean_oos))

    # Build per-event predictions for the incumbent across all CPCV folds and
    # average over folds where each row appears as test. This is a refined
    # CV-averaged prediction matching what the Brain 2 deployment will see.
    pred_sum = np.zeros(n_samples, dtype=np.float64)
    pred_count = np.zeros(n_samples, dtype=np.int64)
    for fold in folds:
        if fold.train_idx.size < MIN_CLASS_COUNT or fold.test_idx.size < MIN_CLASS_COUNT:
            continue
        train_classes = np.bincount(y_arr[fold.train_idx], minlength=2)
        test_classes = np.bincount(y_arr[fold.test_idx], minlength=2)
        if train_classes.min() < MIN_CLASS_COUNT or test_classes.min() < MIN_CLASS_COUNT:
            continue
        est = strategy_factories[incumbent_idx]()
        est.fit(X_arr[fold.train_idx], y_arr[fold.train_idx])
        proba = _positive_class_proba(est, X_arr[fold.test_idx])
        pred_sum[fold.test_idx] += proba
        pred_count[fold.test_idx] += 1
    safe_count = np.where(pred_count > 0, pred_count, 1)
    avg_proba = pred_sum / safe_count
    # Strategy returns from the averaged predictions.
    strategy_returns = _signal_to_returns(avg_proba, r_arr, threshold=bet_threshold)
    dsr_result = deflated_sharpe_ratio(
        strategy_returns,
        n_trials=n_trials,
        sharpe_std_of_trials=sharpe_std_of_trials,
        periods_per_year=periods_per_year,
        min_trials=dsr_min_trials,
    )

    shuffling_result = target_shuffling_test(
        strategy_factories[incumbent_idx],
        X_arr,
        y_arr,
        t0_arr,
        t1_arr,
        n_shuffles=n_shuffles,
        alpha=shuffle_alpha,
        n_splits=max(n_groups - n_test_groups, 2),
        embargo_pct=embargo_pct,
        random_state=random_state,
        raise_on_leakage=raise_on_leakage,
    )

    return ValidationResult(
        pbo=pbo_result,
        dsr=dsr_result,
        target_shuffling=shuffling_result,
        incumbent_idx=incumbent_idx,
        pbo_threshold=pbo_threshold,
        dsr_threshold=dsr_threshold,
    )
