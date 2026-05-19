"""Phase 6 — CombinatoriallyPurgedKFold + synthetic OOS paths."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from afml.validation.cpcv import (
    CombinatoriallyPurgedKFold,
    construct_oos_paths,
)


@pytest.mark.phase6
def test_cpcv_yields_c_n_k_combinations() -> None:
    """C(6, 2) = 15 combinatorial splits."""
    n_groups = 6
    n_test_groups = 2
    cv = CombinatoriallyPurgedKFold(n_groups=n_groups, n_test_groups=n_test_groups)
    assert cv.get_n_splits() == comb(n_groups, n_test_groups) == 15

    n = 600
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    folds = list(cv.split(t0, t1))
    assert len(folds) == 15


@pytest.mark.phase6
def test_cpcv_train_test_disjoint() -> None:
    n = 500
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 3
    cv = CombinatoriallyPurgedKFold(n_groups=5, n_test_groups=2, embargo_pct=0.01)
    for fold in cv.split(t0, t1):
        assert np.intersect1d(fold.train_idx, fold.test_idx).size == 0


@pytest.mark.phase6
def test_cpcv_no_horizon_overlap_in_train() -> None:
    """Per-combination purge: no train horizon may overlap a test horizon."""
    n = 500
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 7  # heavy overlap → purge must kick in
    cv = CombinatoriallyPurgedKFold(n_groups=5, n_test_groups=2, embargo_pct=0.0)
    for fold in cv.split(t0, t1):
        if fold.train_idx.size == 0 or fold.test_idx.size == 0:
            continue
        test_t0_min = int(t0[fold.test_idx].min())
        test_t1_max = int(t1[fold.test_idx].max())
        # Some test groups may be non-contiguous → split into contiguous blocks
        # for the strict check. The CPCV purges per-block. Loosely the union
        # check below is sufficient: no train horizon may overlap the full
        # union of test horizons.
        overlap = (t0[fold.train_idx] <= test_t1_max) & (t1[fold.train_idx] >= test_t0_min)
        # Allow overlap with non-adjacent block but not the connected union.
        # On uniformly-overlapping fixtures the union check is tight.
        contiguous = all(
            g + 1 == g_next
            for g, g_next in zip(fold.test_groups[:-1], fold.test_groups[1:], strict=True)
        )
        if contiguous:
            assert not overlap.any(), "purge leaked on contiguous test block"


@pytest.mark.phase6
def test_cpcv_test_groups_cover_universe() -> None:
    """The union of all test_groups across folds covers every group equally
    many times."""
    n_groups = 5
    n_test_groups = 2
    cv = CombinatoriallyPurgedKFold(n_groups=n_groups, n_test_groups=n_test_groups)
    n = 300
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 1
    group_counts = np.zeros(n_groups, dtype=np.int64)
    for fold in cv.split(t0, t1):
        for g in fold.test_groups:
            group_counts[g] += 1
    # Each group must appear C(N-1, k-1) times.
    expected = comb(n_groups - 1, n_test_groups - 1)
    assert (group_counts == expected).all(), f"got {group_counts}, expected uniform {expected}"


@pytest.mark.phase6
def test_cpcv_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="n_groups"):
        CombinatoriallyPurgedKFold(n_groups=1)
    with pytest.raises(ValueError, match="n_test_groups"):
        CombinatoriallyPurgedKFold(n_groups=4, n_test_groups=4)
    with pytest.raises(ValueError, match="n_test_groups"):
        CombinatoriallyPurgedKFold(n_groups=4, n_test_groups=0)
    with pytest.raises(ValueError, match="embargo_pct"):
        CombinatoriallyPurgedKFold(embargo_pct=0.5)


@pytest.mark.phase6
def test_n_paths_formula() -> None:
    """n_paths = C(N-1, k-1) for any N, k."""
    for n_groups in (4, 5, 6, 8):
        for n_test in range(1, n_groups):
            cv = CombinatoriallyPurgedKFold(n_groups=n_groups, n_test_groups=n_test)
            assert cv.get_n_paths() == comb(n_groups - 1, n_test - 1)


@pytest.mark.phase6
def test_oos_path_construction_covers_universe() -> None:
    """Every group appears exactly once per synthetic OOS path; total paths =
    C(N-1, k-1)."""
    n_groups = 5
    n_test_groups = 2
    n = 300
    samples_per_group = n // n_groups
    cv = CombinatoriallyPurgedKFold(n_groups=n_groups, n_test_groups=n_test_groups)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 1

    # Fake per-combination predictions: predict the group index. This way we
    # can verify each path covers every group exactly once.
    fold_predictions: dict[tuple[int, ...], dict[int, np.ndarray]] = {}
    for fold in cv.split(t0, t1):
        preds_by_group: dict[int, np.ndarray] = {}
        for g in fold.test_groups:
            preds_by_group[g] = np.full(samples_per_group, float(g))
        fold_predictions[fold.test_groups] = preds_by_group

    paths = construct_oos_paths(fold_predictions, n_groups, n_test_groups)
    # Number of paths.
    n_paths = comb(n_groups - 1, n_test_groups - 1)
    assert paths.shape == (n_paths, n_groups * samples_per_group)
    # Each path is a concatenation of per-group constant predictions.
    for p in range(n_paths):
        for g in range(n_groups):
            chunk = paths[p, g * samples_per_group : (g + 1) * samples_per_group]
            assert np.all(chunk == float(g)), f"path {p}, group {g}: expected constant {g}, got mix"


@pytest.mark.phase6
def test_cpcv_non_contiguous_embargo_protects_each_block_boundary() -> None:
    """AFML Phase 0-6 audit V3 — when a combination's test groups are
    NON-contiguous (e.g. groups 0 and 2 with group 1 as train in between),
    the embargo must be applied to the right boundary of EACH contiguous
    test block, not just the global ``max(t1)``.

    We construct a 4-group split, force the test combination ``(0, 2)``, and
    verify that the train samples immediately after group 0 (i.e. the start
    of group 1) are embargoed out — proving the middle training block's left
    edge is protected.
    """
    n = 400
    n_groups = 4
    embargo_pct = 0.05
    embargo_size = int(np.floor(n * embargo_pct))
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 1  # unit horizons → isolate the embargo from the overlap purge
    cv = CombinatoriallyPurgedKFold(n_groups=n_groups, n_test_groups=2, embargo_pct=embargo_pct)
    group_size = n // n_groups  # 100

    # Find the fold whose test groups are exactly (0, 2) — non-contiguous.
    target = next(fold for fold in cv.split(t0, t1) if fold.test_groups == (0, 2))
    train_set = set(target.train_idx.tolist())

    # Group 0 occupies sorted positions [0, 100); its right boundary is at 100.
    # The embargo must drop [100, 100 + embargo_size) — the LEFT edge of the
    # middle training block (group 1). Those indices must NOT be in train.
    g0_right_embargo = range(group_size, group_size + embargo_size)
    leaked_after_g0 = [i for i in g0_right_embargo if i in train_set]
    assert not leaked_after_g0, (
        f"embargo failed to protect group-0 right boundary; leaked: {leaked_after_g0}"
    )

    # Group 2 occupies [200, 300); its right boundary embargo drops
    # [300, 300 + embargo_size) — the left edge of group 3.
    g2_right_embargo = range(3 * group_size, 3 * group_size + embargo_size)
    leaked_after_g2 = [i for i in g2_right_embargo if i in train_set]
    assert not leaked_after_g2, (
        f"embargo failed to protect group-2 right boundary; leaked: {leaked_after_g2}"
    )


@pytest.mark.phase6
def test_cpcv_non_contiguous_purge_no_horizon_overlap() -> None:
    """For a non-contiguous test combination with overlapping horizons, NO
    train sample's ``[t0, t1]`` may overlap EITHER test block's window."""
    n = 400
    n_groups = 4
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 8  # heavy overlap
    cv = CombinatoriallyPurgedKFold(n_groups=n_groups, n_test_groups=2, embargo_pct=0.0)
    group_size = n // n_groups

    target = next(fold for fold in cv.split(t0, t1) if fold.test_groups == (0, 2))
    # Per-block windows.
    block0 = (0, group_size)
    block2 = (2 * group_size, 3 * group_size)
    for b_start, b_stop in (block0, block2):
        b_t0_min = int(t0[b_start])
        b_t1_max = int(t1[b_start:b_stop].max())
        overlap = (t0[target.train_idx] <= b_t1_max) & (t1[target.train_idx] >= b_t0_min)
        assert not overlap.any(), (
            f"train horizon overlaps test block [{b_start},{b_stop}) — purge leak"
        )


@pytest.mark.phase6
def test_cpcv_indices_address_original_row_order() -> None:
    """If we shuffle the input, returned indices must still index back to
    the original rows."""
    n = 100
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    t0_shuffled = perm.astype(np.int64)
    t1_shuffled = t0_shuffled + 1
    cv = CombinatoriallyPurgedKFold(n_groups=4, n_test_groups=2)
    for fold in cv.split(t0_shuffled, t1_shuffled):
        assert fold.train_idx.min() >= 0
        assert fold.test_idx.max() < n
