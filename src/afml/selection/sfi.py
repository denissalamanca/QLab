"""Single Feature Importance (SFI) — Clustered MDA fallback.

Activated by the Phase 4 orchestrator when Clustered MDA fails to reduce the
feature count by ≥ 20 % (see ``afml.selection.pipeline``).

Per López de Prado 2018 §8.4.4, SFI fits a *separate* classifier with each
feature in isolation and measures the OOS NLL improvement over the no-feature
baseline (intercept-only / class-prior). Features that beat the baseline by a
material margin survive; others are dropped.

This is genuinely orthogonal to Clustered MDA: MDA tests "what if I take this
group away from a full model?"; SFI tests "what can this feature alone explain?".
Either alone is incomplete, but together they form a reliable two-arm fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier

from afml.selection.mda import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_N_ESTIMATORS,
    MIN_CLASS_COUNT,
    _predict_proba_positive,
    _safe_log_loss,
)
from afml.selection.purged_kfold import PurgedKFold

# Per-feature NLL improvement threshold (in nats). A feature must drop the NLL
# by at least this much vs the class-prior baseline on average to survive.
DEFAULT_MIN_NLL_IMPROVEMENT: float = 1e-3


@dataclass(frozen=True, slots=True)
class SFIResult:
    """Output of :func:`single_feature_importance`."""

    feature_importance: dict[int, float]  # column_idx → mean NLL improvement
    feature_importance_std: dict[int, float]
    surviving_columns: list[int]
    baseline_nll_per_fold: npt.NDArray[np.float64]


def single_feature_importance(
    X: npt.NDArray[np.floating],
    y: npt.NDArray[np.integer],
    t0: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    t1: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    *,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    min_nll_improvement: float = DEFAULT_MIN_NLL_IMPROVEMENT,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    random_state: int = 0,
) -> SFIResult:
    """Rank features by OOS NLL improvement over class-prior baseline.

    For each feature ``j``, fit a ``RandomForestClassifier`` on column ``j``
    alone in each Purged K-Fold train fold; compute the NLL on the test fold.
    The class-prior baseline NLL — what you'd get with a constant prediction
    ``p̂ = mean(y_train)`` — is the reference.

    Improvement is ``baseline_nll − feature_nll`` (positive means the feature
    is helpful).
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X_arr.shape}")
    n_samples, n_features = X_arr.shape
    if y_arr.shape != (n_samples,):
        raise ValueError(f"y shape {y_arr.shape} != (n_samples,) {(n_samples,)}")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X must be finite")
    if not set(np.unique(y_arr).tolist()) <= {0, 1}:
        raise ValueError("y must be binary in {0, 1}")

    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    folds = list(cv.split(t0, t1))
    n_folds = len(folds)

    baseline_nll_per_fold = np.full(n_folds, np.nan, dtype=np.float64)
    feature_nll = np.full((n_folds, n_features), np.nan, dtype=np.float64)

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        # Class-prior baseline: predict mean(y_train) for every test row.
        train_counts = np.bincount(y_train, minlength=2)
        if train_counts.min() < MIN_CLASS_COUNT or len(np.unique(y_test)) < MIN_CLASS_COUNT:
            # Degenerate fold — leave NaN; will be filtered in the aggregation.
            continue
        prior = float(train_counts[1] / train_counts.sum())
        baseline_proba = np.full(X_test.shape[0], prior, dtype=np.float64)
        baseline_nll_per_fold[fold_i] = _safe_log_loss(y_test, baseline_proba)

        for j in range(n_features):
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state + fold_i,
                n_jobs=1,
            )
            clf.fit(X_train[:, [j]], y_train)
            proba = _predict_proba_positive(clf, X_test[:, [j]])
            feature_nll[fold_i, j] = _safe_log_loss(y_test, proba)

    feature_importance: dict[int, float] = {}
    feature_importance_std: dict[int, float] = {}
    surviving_columns: list[int] = []

    for j in range(n_features):
        improvements = baseline_nll_per_fold - feature_nll[:, j]
        valid = improvements[~np.isnan(improvements)]
        if valid.size == 0:
            mean_imp = 0.0
            std_imp = 0.0
        else:
            mean_imp = float(valid.mean())
            std_imp = float(valid.std(ddof=1)) if valid.size > 1 else 0.0
        feature_importance[j] = mean_imp
        feature_importance_std[j] = std_imp
        if mean_imp >= min_nll_improvement:
            surviving_columns.append(j)

    return SFIResult(
        feature_importance=feature_importance,
        feature_importance_std=feature_importance_std,
        surviving_columns=sorted(surviving_columns),
        baseline_nll_per_fold=baseline_nll_per_fold,
    )
