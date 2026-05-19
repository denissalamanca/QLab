"""Sequentially Bootstrapped Random Forest (SBRF) — Blueprint §7.2.

Why we need a custom ensemble instead of using ``RandomForestClassifier`` with
``sample_weight``:

- Standard ``RandomForestClassifier`` performs a *uniform* bootstrap inside
  each tree, even when ``sample_weight`` is non-uniform. The weights bias the
  *split criterion* but not the *resampling*. On overlapping labels that is
  insufficient: clustered events still get redrawn together at uniform odds.
- The Blueprint mandates the sequential bootstrap as the resampling
  distribution itself. We therefore subclass ``BaseEstimator`` and roll our
  own per-tree fit on a sequentially-bootstrapped subsample.

What we DO retain from sklearn:

- ``DecisionTreeClassifier`` as the base learner (so feature importances,
  splitting criteria, etc. stay battle-tested).
- ``sample_weight`` on each per-tree ``fit`` is set to the **average
  uniqueness** of the drawn samples. This applies the second-order
  AFML correction *within* a tree (Snippet 4.10).

Prediction: average ``predict_proba`` across trees, identical to a vanilla RF.
Calibration is handled by :mod:`afml.modeling.calibration` post-hoc.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from afml.modeling.concurrency import average_uniqueness
from afml.modeling.sequential_bootstrap import sequential_bootstrap

DEFAULT_N_ESTIMATORS: int = 200
DEFAULT_MAX_DEPTH: int = 5
DEFAULT_MIN_SAMPLES_LEAF: int = 5
DEFAULT_BOOTSTRAP_FRACTION: float = 1.0
DEFAULT_MAX_FEATURES: str = "sqrt"


class SequentiallyBootstrappedRandomForest(BaseEstimator, ClassifierMixin):
    """Random forest where each tree is fit on a sequentially-bootstrapped sample.

    Parameters
    ----------
    n_estimators
        Number of trees.
    max_depth
        Per-tree maximum depth. ``None`` = grow until pure / min_samples_leaf.
    min_samples_leaf
        Minimum samples per leaf.
    max_features
        Per-split feature subset size. Accepts the same values as
        ``sklearn.tree.DecisionTreeClassifier``.
    bootstrap_fraction
        Fraction of ``n_samples`` to draw per tree. ``1.0`` matches sklearn's
        default. Lower values give faster fits at the cost of variance.
    random_state
        Master seed. Each tree's draws + fit are reseeded from
        ``random_state + tree_index``.
    n_jobs
        Currently single-threaded; the parameter is here for forward
        compatibility (the per-tree fits are embarrassingly parallel).

    Attributes
    ----------
    estimators_
        List of fitted ``DecisionTreeClassifier`` instances.
    classes_
        Discovered classes from ``y`` at fit time (always ``[0, 1]`` for
        binary meta-labelling but exposed for general compatibility).
    n_classes_
        Length of ``classes_``.
    n_features_in_
        ``X.shape[1]`` at fit time.
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        *,
        max_depth: int | None = DEFAULT_MAX_DEPTH,
        min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
        max_features: str | int | float | None = DEFAULT_MAX_FEATURES,
        bootstrap_fraction: float = DEFAULT_BOOTSTRAP_FRACTION,
        random_state: int | None = 0,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap_fraction = bootstrap_fraction
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(
        self,
        X: npt.NDArray[np.floating],
        y: npt.NDArray[np.integer],
        indicator_mat: npt.NDArray[np.integer],
        *,
        sample_weight: npt.NDArray[np.floating] | None = None,
    ) -> SequentiallyBootstrappedRandomForest:
        """Fit ``n_estimators`` trees.

        Parameters
        ----------
        X
            ``(n_samples, n_features)`` feature matrix.
        y
            ``(n_samples,)`` binary labels.
        indicator_mat
            ``(n_grid, n_samples)`` indicator matrix — columns ALIGNED with X
            rows. Built via :func:`afml.modeling.indicator_matrix` on the
            event horizons.
        sample_weight
            Optional per-sample weights (in addition to the per-tree
            ``ū``-based weights). ``None`` defaults to uniform.

        Returns
        -------
        ``self``.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        ind = np.asarray(indicator_mat, dtype=np.int64)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X_arr.shape}")
        n_samples, n_features = X_arr.shape
        if y_arr.shape != (n_samples,):
            raise ValueError(f"y shape {y_arr.shape} != (n_samples,) {(n_samples,)}")
        if ind.ndim != 2 or ind.shape[1] != n_samples:
            raise ValueError(
                f"indicator_mat shape {ind.shape} must be (n_grid, n_samples={n_samples})"
            )
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64)
            if sw.shape != (n_samples,):
                raise ValueError(f"sample_weight shape {sw.shape} != ({n_samples},)")
        else:
            sw = np.ones(n_samples, dtype=np.float64)

        bootstrap_size = max(1, round(self.bootstrap_fraction * n_samples))
        master_rng = np.random.default_rng(self.random_state)

        estimators: list[DecisionTreeClassifier] = []
        for tree_i in range(self.n_estimators):
            tree_rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1) ^ tree_i)
            draw = sequential_bootstrap(ind, n_samples=bootstrap_size, rng=tree_rng)

            X_boot = X_arr[draw]
            y_boot = y_arr[draw]
            sw_boot = sw[draw]
            # Also incorporate average uniqueness as a tree-level sample_weight
            # multiplier (AFML Snippet 4.10): events that appear multiple times
            # in the draw already have a multiplicity bias, so we weight by ū.
            ind_boot = ind[:, draw]
            avg_u_boot = average_uniqueness(ind_boot)
            tree_weight = sw_boot * avg_u_boot
            # Sklearn requires strictly positive sample_weight; clip tiny
            # floor to avoid degenerate splits.
            tree_weight = np.maximum(tree_weight, 1e-12)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(tree_rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X_boot, y_boot, sample_weight=tree_weight)
            estimators.append(tree)

        self.estimators_ = estimators
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.n_classes_ = 2
        self.n_features_in_ = n_features
        return self

    def predict_proba(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """Mean of per-tree ``predict_proba``.

        Returns ``(n_samples, 2)`` array with columns ``[P(y=0), P(y=1)]``.
        Trees that saw only one class during fit emit a degenerate
        ``predict_proba`` (single column); we re-expand them to ``[0, 0]``
        / ``[0, 1]`` based on their ``classes_`` so the aggregation is sound.
        """
        check_is_fitted(self, "estimators_")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X_arr.shape}")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, fit on {self.n_features_in_}")

        n = X_arr.shape[0]
        accum = np.zeros((n, 2), dtype=np.float64)
        for tree in self.estimators_:
            proba = tree.predict_proba(X_arr)
            classes = np.asarray(tree.classes_)
            if proba.shape[1] == 1:
                # Single-class tree: assign all mass to that class.
                col = int(np.where(self.classes_ == int(classes[0]))[0][0])
                accum[:, col] += proba[:, 0]
            else:
                for tree_col, cls in enumerate(classes):
                    out_col = int(np.where(self.classes_ == int(cls))[0][0])
                    accum[:, out_col] += proba[:, tree_col]
        return accum / float(len(self.estimators_))

    def predict(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.int64]:
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)], dtype=np.int64)

    def __sklearn_tags__(self) -> Any:
        """Advertise this estimator as a classifier so sklearn's helpers
        (``CalibratedClassifierCV``, ``is_classifier``, ``FrozenEstimator``,
        etc.) treat it correctly.

        sklearn ≥ 1.6 introduced the ``__sklearn_tags__`` protocol; in earlier
        versions the legacy ``_estimator_type = "classifier"`` was used. We
        keep both for forward / backward compat.
        """
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    # Legacy fallback for callers that still inspect _estimator_type directly.
    _estimator_type: str = "classifier"
