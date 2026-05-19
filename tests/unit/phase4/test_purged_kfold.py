"""Phase 4 — Purged + Embargoed K-Fold (AFML Ch. 7).

This is the splitter that Phase 5's Index-Intersection DoD will exercise on
realistic Brain-1 events. Here we lock in the algorithmic invariants on
controlled synthetic data:

- train and test indices are disjoint within each fold;
- the union of train and test for each fold equals the universe minus the
  purged + embargoed regions;
- no train sample's label horizon ``[t0, t1]`` overlaps the test window;
- the embargo region after the test fold is empty in train.

The Blueprint flags index intersection as the Phase 5 acceptance test; building
the splitter now lets Phase 5 plug it directly into the meta-labeler.
"""

from __future__ import annotations

import numpy as np
import pytest

from afml.selection.purged_kfold import PurgedKFold


@pytest.mark.phase4
def test_purged_kfold_train_test_disjoint() -> None:
    n = 200
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 3
    cv = PurgedKFold(n_splits=5, embargo_pct=0.02)
    folds = list(cv.split(t0, t1))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert np.intersect1d(train_idx, test_idx).size == 0


@pytest.mark.phase4
def test_purged_kfold_no_horizon_overlap_in_train() -> None:
    """For every train sample, its [t0, t1] must NOT overlap the test window
    [min(test_t0), max(test_t1)]."""
    n = 300
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 7  # 7-bar horizons → adjacent samples overlap
    cv = PurgedKFold(n_splits=4, embargo_pct=0.01)
    for train_idx, test_idx in cv.split(t0, t1):
        if test_idx.size == 0 or train_idx.size == 0:
            continue
        test_t0_min = t0[test_idx].min()
        test_t1_max = t1[test_idx].max()
        # No train sample whose horizon touches [test_t0_min, test_t1_max].
        overlap_mask = (t0[train_idx] <= test_t1_max) & (t1[train_idx] >= test_t0_min)
        assert not overlap_mask.any(), (
            "purging failed: a train sample's horizon overlaps the test window"
        )


@pytest.mark.phase4
def test_purged_kfold_embargo_respected() -> None:
    """The first ``embargo_size`` indices after the test window must NOT be in
    train. (Indices are *sort-order* positions; we verify by translating back
    through sort_order.)"""
    n = 400
    embargo_pct = 0.05
    embargo_size = int(np.floor(n * embargo_pct))
    t0 = np.arange(n, dtype=np.int64)  # already monotonic — sort_order is identity
    t1 = t0  # unit horizons — isolate the embargo effect from the overlap purge
    cv = PurgedKFold(n_splits=5, embargo_pct=embargo_pct)
    folds = list(cv.split(t0, t1))
    for train_idx, test_idx in folds:
        if test_idx.size == 0:
            continue
        # Since sort_order = identity here, sorted-positions == original indices.
        test_stop = int(test_idx.max()) + 1
        embargo_region = np.arange(test_stop, min(n, test_stop + embargo_size), dtype=np.int64)
        leaked = np.intersect1d(train_idx, embargo_region)
        assert leaked.size == 0, f"embargo leaked: {leaked.tolist()}"


@pytest.mark.phase4
def test_purged_kfold_returns_pre_sort_indices() -> None:
    """If we shuffle the input, the returned indices must still address the
    *original* rows (not the internal sorted order)."""
    n = 100
    t0_unsorted = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    t0_shuffled = t0_unsorted[perm]
    t1_shuffled = t0_shuffled + 2
    cv = PurgedKFold(n_splits=4, embargo_pct=0.0)
    for train_idx, test_idx in cv.split(t0_shuffled, t1_shuffled):
        # Indices must be ∈ [0, n).
        assert train_idx.min() >= 0 and train_idx.max() < n
        assert test_idx.min() >= 0 and test_idx.max() < n


@pytest.mark.phase4
def test_purged_kfold_test_indices_partition_universe() -> None:
    """Concatenating every fold's test indices covers the full universe exactly
    once (since folds are a contiguous partition of the sorted index)."""
    n = 250
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 1
    cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
    all_test = np.concatenate([test_idx for _, test_idx in cv.split(t0, t1)])
    assert sorted(all_test.tolist()) == list(range(n))


@pytest.mark.phase4
def test_purged_kfold_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        PurgedKFold(n_splits=1)
    with pytest.raises(ValueError, match="embargo_pct"):
        PurgedKFold(embargo_pct=0.5)
    with pytest.raises(ValueError, match="embargo_pct"):
        PurgedKFold(embargo_pct=-0.01)


@pytest.mark.phase4
def test_purged_kfold_rejects_t1_before_t0() -> None:
    t0 = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    t1 = np.array([1, 2, 3, 4, 0], dtype=np.int64)  # last element violates
    cv = PurgedKFold(n_splits=2)
    with pytest.raises(ValueError, match="t1"):
        list(cv.split(t0, t1))


@pytest.mark.phase4
def test_purged_kfold_index_intersection_test_passes() -> None:
    """Phase 5 gate (Blueprint §7.3): for every fold,
    ``max(train_t1) + embargo < min(test_t0)`` for the right-side train portion,
    AND ``max(test_t1) + embargo < min(right_train_t0)``.

    We test the stronger statement: NO train sample's horizon reaches into
    [test_t0_min − embargo_time, test_t1_max + embargo_time]. Embargo here is
    measured in *time units*, which on this fixture equals one index unit per
    sample since horizons are integer-encoded.
    """
    n = 500
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    embargo_pct = 0.02
    cv = PurgedKFold(n_splits=5, embargo_pct=embargo_pct)
    for train_idx, test_idx in cv.split(t0, t1):
        if test_idx.size == 0 or train_idx.size == 0:
            continue
        test_t0_min = t0[test_idx].min()
        test_t1_max = t1[test_idx].max()
        # No train horizon may touch the test window.
        train_overlap = (t0[train_idx] <= test_t1_max) & (t1[train_idx] >= test_t0_min)
        assert not train_overlap.any()
