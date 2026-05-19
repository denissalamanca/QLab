"""Clustered Mean Decrease Accuracy via Purged K-Fold (Blueprint §6.2).

This replaces sklearn's MDI / Gini ``feature_importances_`` which are **banned**
across this codebase (CLAUDE.md anti-shortcut rule). MDI overstates the
importance of high-cardinality / continuous features and is computed on the
training set — both bias the result.

The algorithm:

1. Fit a probabilistic classifier on each ``train_idx`` from
   :class:`afml.selection.PurgedKFold`.
2. Score the baseline OOS negative log-loss on ``test_idx``.
3. For every cluster, **simultaneously permute** all features belonging to it
   on the test rows, re-score, and record the NLL inflation
   ``imp_c = nll_perm − nll_base``.
4. Average ``imp_c`` over folds. Significance via a paired one-sided t-test
   over the per-fold importances (``H_0 : E[imp_c] = 0``, ``H_1 : E[imp_c] > 0``).
   Clusters with ``p ≤ significance_threshold`` are kept; the rest are dropped.

The "shuffle every feature in the cluster simultaneously" detail is what makes
this *clustered* MDA: collinear features cancel each other's permutations under
plain per-feature MDA (you shuffle X1 but X2 is still informative), so collinear
groups never look important. Clustering them and shuffling jointly closes that
hole — exactly the Blueprint requirement.

We use ``sklearn.ensemble.RandomForestClassifier`` strictly for ``predict_proba``.
``.feature_importances_`` is never read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from afml.selection.purged_kfold import PurgedKFold

DEFAULT_N_ESTIMATORS: int = 200
DEFAULT_MAX_DEPTH: int = 5
DEFAULT_MIN_SAMPLES_LEAF: int = 5
DEFAULT_SIGNIFICANCE_THRESHOLD: float = 0.05
# Floor on predict_proba to avoid log(0).
PROBA_EPSILON: float = 1e-6
# Minimum class diversity in a training fold — below this, RandomForest
# emits a warning and predict_proba returns a degenerate column.
MIN_CLASS_COUNT: int = 2


@dataclass(frozen=True, slots=True)
class ClusteredMDAResult:
    """Output of :func:`clustered_mda`.

    Attributes
    ----------
    cluster_importance
        ``cluster_id → mean NLL inflation`` averaged across folds.
    cluster_importance_std
        Per-cluster standard deviation across folds.
    cluster_pvalue
        Per-cluster one-sided t-test p-value against ``H_0 : importance = 0``.
    kept_clusters
        Cluster IDs whose ``pvalue ≤ significance_threshold`` AND
        ``cluster_importance > 0`` (positive degradation under permutation).
    fold_importances
        ``(n_folds, n_clusters)`` raw matrix — useful for downstream
        bootstrapped error bars.
    """

    cluster_importance: dict[int, float]
    cluster_importance_std: dict[int, float]
    cluster_pvalue: dict[int, float]
    kept_clusters: list[int]
    fold_importances: npt.NDArray[np.float64]


def _safe_log_loss(
    y_true: npt.NDArray[np.int64],
    y_prob_pos: npt.NDArray[np.float64],
) -> float:
    """NLL with epsilon-clipping for numerical safety.

    ``y_prob_pos`` is the probability of class 1 (binary classification).
    """
    p = np.clip(y_prob_pos, PROBA_EPSILON, 1.0 - PROBA_EPSILON)
    return float(log_loss(y_true, p, labels=[0, 1]))


def _predict_proba_positive(
    classifier: RandomForestClassifier,
    X_test: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Robustly extract P(y=1 | X) even from degenerate single-class fits."""
    proba = classifier.predict_proba(X_test)
    classes = classifier.classes_
    if proba.shape[1] == 1:
        # Single-class training fold: probabilities all map to the trained class.
        only_class = int(classes[0])
        return np.full(X_test.shape[0], 1.0 if only_class == 1 else 0.0, dtype=np.float64)
    pos_col = int(np.where(classes == 1)[0][0])
    return np.asarray(proba[:, pos_col], dtype=np.float64)


def clustered_mda(  # noqa: PLR0912, PLR0915 — single linear bookkeeping loop; splitting hurts clarity
    X: npt.NDArray[np.floating],
    y: npt.NDArray[np.integer],
    cluster_labels: npt.NDArray[np.integer],
    t0: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    t1: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    *,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    significance_threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    random_state: int = 0,
) -> ClusteredMDAResult:
    """Compute Clustered MDA importances via Purged K-Fold.

    Parameters
    ----------
    X
        ``(n_samples, n_features)`` feature matrix. No NaN / Inf.
    y
        ``(n_samples,)`` binary labels in ``{0, 1}``.
    cluster_labels
        ``(n_features,)`` cluster IDs from :func:`cluster_features_onc`.
        Cluster IDs need not be contiguous; we work with whatever set is
        present.
    t0, t1
        Per-sample label horizons (see :class:`PurgedKFold`).
    n_splits, embargo_pct
        Forwarded to :class:`PurgedKFold`.
    significance_threshold
        Maximum acceptable one-sided p-value for keeping a cluster.
    n_estimators, max_depth, min_samples_leaf
        RandomForest hyperparameters. Defaults are conservative — shallow trees
        and many estimators minimise variance in the permutation-importance
        estimate.
    random_state
        Seeds both the RandomForest and the permutation RNG so the test is
        reproducible.

    Returns
    -------
    :class:`ClusteredMDAResult`.

    Raises
    ------
    ValueError
        Input shape mismatches, non-binary ``y``, or non-finite ``X``.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    cluster_arr = np.asarray(cluster_labels, dtype=np.int64)

    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X_arr.shape}")
    n_samples, n_features = X_arr.shape
    if y_arr.shape != (n_samples,):
        raise ValueError(f"y shape {y_arr.shape} != (n_samples,) {(n_samples,)}")
    if cluster_arr.shape != (n_features,):
        raise ValueError(
            f"cluster_labels shape {cluster_arr.shape} != (n_features,) {(n_features,)}"
        )
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X must be finite — Phase 3 must drop NaN / Inf first")
    if not set(np.unique(y_arr).tolist()) <= {0, 1}:
        raise ValueError(f"y must be binary in {{0, 1}}, got {set(np.unique(y_arr).tolist())}")

    unique_clusters = sorted({int(c) for c in cluster_arr})
    n_clusters = len(unique_clusters)
    cluster_to_col_idx: dict[int, npt.NDArray[np.int64]] = {
        c: np.where(cluster_arr == c)[0].astype(np.int64) for c in unique_clusters
    }

    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    folds = list(cv.split(t0, t1))
    n_folds = len(folds)

    fold_importances = np.full((n_folds, n_clusters), np.nan, dtype=np.float64)
    rng = np.random.default_rng(random_state)

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        # Degenerate folds — skip but emit zero importance so the fold doesn't
        # warp the t-test toward a false positive.
        train_class_counts = np.bincount(y_train, minlength=2)
        if train_class_counts.min() < MIN_CLASS_COUNT or len(np.unique(y_test)) < MIN_CLASS_COUNT:
            fold_importances[fold_i, :] = 0.0
            continue

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + fold_i,
            n_jobs=1,
        )
        clf.fit(X_train, y_train)

        proba_base = _predict_proba_positive(clf, X_test)
        nll_base = _safe_log_loss(y_test, proba_base)

        for cluster_i, cluster_id in enumerate(unique_clusters):
            cols = cluster_to_col_idx[cluster_id]
            X_test_perm = X_test.copy()
            # Simultaneously permute every column in the cluster, using the
            # same permutation across all columns to preserve any intra-cluster
            # correlation structure (we want to break correlation with y, not
            # with intra-cluster siblings).
            perm = rng.permutation(X_test.shape[0])
            X_test_perm[:, cols] = X_test[perm][:, cols]

            proba_perm = _predict_proba_positive(clf, X_test_perm)
            nll_perm = _safe_log_loss(y_test, proba_perm)
            fold_importances[fold_i, cluster_i] = nll_perm - nll_base

    # Aggregate.
    cluster_importance: dict[int, float] = {}
    cluster_importance_std: dict[int, float] = {}
    cluster_pvalue: dict[int, float] = {}
    kept_clusters: list[int] = []

    for cluster_i, cluster_id in enumerate(unique_clusters):
        col = fold_importances[:, cluster_i]
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            mean_imp = 0.0
            std_imp = 0.0
            p_val = 1.0
        else:
            mean_imp = float(valid.mean())
            std_imp = float(valid.std(ddof=1)) if valid.size > 1 else 0.0
            if std_imp == 0.0:
                # All folds reported identical importance — degenerate t-stat.
                # If importance is positive give it a free pass; if zero or
                # negative reject.
                p_val = 0.0 if mean_imp > 0 else 1.0
            else:
                # One-sided H_1 : mean > 0.
                t_stat, two_sided_p = stats.ttest_1samp(valid, popmean=0.0)
                p_val = float(two_sided_p / 2.0 if t_stat > 0 else 1.0 - two_sided_p / 2.0)

        cluster_importance[cluster_id] = mean_imp
        cluster_importance_std[cluster_id] = std_imp
        cluster_pvalue[cluster_id] = p_val

        if p_val <= significance_threshold and mean_imp > 0.0:
            kept_clusters.append(cluster_id)

    return ClusteredMDAResult(
        cluster_importance=cluster_importance,
        cluster_importance_std=cluster_importance_std,
        cluster_pvalue=cluster_pvalue,
        kept_clusters=sorted(kept_clusters),
        fold_importances=fold_importances,
    )
