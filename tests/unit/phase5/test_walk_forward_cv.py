"""Phase 5 — PurgedWalkForwardCV invariants.

The Blueprint §7.3 Index-Intersection DoD assumes a one-sided / walk-forward
splitter:

    max(train_t1[train_idx]) + embargo_time < min(test_t0[test_idx])

This file locks in the per-fold invariants. The full DoD assertion across the
end-to-end pipeline is in ``test_pipeline.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from afml.selection import PurgedWalkForwardCV


@pytest.mark.phase5
def test_walk_forward_train_is_chronologically_before_test() -> None:
    n = 200
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 3
    cv = PurgedWalkForwardCV(n_splits=4, embargo_pct=0.02, train_fraction=0.3)
    for train_idx, test_idx in cv.split(t0, t1):
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        assert t0[train_idx].max() < t0[test_idx].min()


@pytest.mark.phase5
def test_walk_forward_index_intersection_holds_per_fold() -> None:
    """Blueprint §7.3: ``max(train_t1) + embargo < min(test_t0)``."""
    n = 500
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 5
    embargo_pct = 0.02
    embargo_size = int(np.floor(n * embargo_pct))
    cv = PurgedWalkForwardCV(n_splits=5, embargo_pct=embargo_pct)
    for train_idx, test_idx in cv.split(t0, t1):
        if train_idx.size == 0:
            continue
        train_t1_max = int(t1[train_idx].max())
        test_t0_min = int(t0[test_idx].min())
        assert train_t1_max + embargo_size < test_t0_min, (
            f"index intersection violated: train_t1_max={train_t1_max}, "
            f"embargo={embargo_size}, test_t0_min={test_t0_min}"
        )


@pytest.mark.phase5
def test_walk_forward_disjoint_train_test() -> None:
    n = 300
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    cv = PurgedWalkForwardCV(n_splits=4, embargo_pct=0.02)
    for train_idx, test_idx in cv.split(t0, t1):
        assert np.intersect1d(train_idx, test_idx).size == 0


@pytest.mark.phase5
def test_walk_forward_burn_in_respected() -> None:
    """No test sample should fall inside the burn-in region."""
    n = 200
    burn_in_fraction = 0.4
    burn_in_size = int(np.floor(n * burn_in_fraction))
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 1
    cv = PurgedWalkForwardCV(n_splits=3, embargo_pct=0.0, train_fraction=burn_in_fraction)
    for _, test_idx in cv.split(t0, t1):
        assert int(t0[test_idx].min()) >= burn_in_size


@pytest.mark.phase5
def test_walk_forward_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        PurgedWalkForwardCV(n_splits=0)
    with pytest.raises(ValueError, match="embargo_pct"):
        PurgedWalkForwardCV(embargo_pct=0.5)
    with pytest.raises(ValueError, match="train_fraction"):
        PurgedWalkForwardCV(train_fraction=1.5)


@pytest.mark.phase5
def test_walk_forward_purges_long_horizon_overhang() -> None:
    """A train sample with a long horizon ``[t0, t1]`` that reaches into the
    test window must be purged even though it sits well before the test."""
    n = 100
    t0 = np.arange(n, dtype=np.int64)
    # Make one early sample's t1 reach far into the future.
    t1 = t0 + 1
    t1[10] = 80  # sample 10 has horizon [10, 80] — overhangs into test folds
    cv = PurgedWalkForwardCV(n_splits=3, embargo_pct=0.0, train_fraction=0.3)
    for train_idx, test_idx in cv.split(t0, t1):
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        test_t0_min = int(t0[test_idx].min())
        long_horizon_t1 = 80
        long_horizon_t0 = 10
        if long_horizon_t0 < test_t0_min <= long_horizon_t1:
            # That sample overlaps this fold — must NOT be in train.
            assert long_horizon_t0 not in train_idx
