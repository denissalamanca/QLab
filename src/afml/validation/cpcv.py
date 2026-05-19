"""Combinatorially Purged Cross-Validation (López de Prado 2018 Ch. 12).

Standard k-fold and even purged-k-fold yield a single OOS sequence per
sample. CPCV generates **multiple** synthetic OOS paths from the same data,
giving us a *distribution* of strategy performances rather than a point
estimate. That distribution is what the Bailey-Lopez de Prado PBO statistic
operates on.

Algorithm:

1. Divide the sample index (sorted by ``t0``) into ``N`` contiguous groups.
2. For every combination of ``k`` test-groups out of ``N``: ``C(N, k)`` total.
3. Train on the remaining ``N - k`` groups, applying:
   - **Purge** — drop train samples whose label horizon ``[t0, t1]`` overlaps
     the test window.
   - **Embargo** — drop ``embargo_pct × n`` samples immediately after each
     contiguous test block (autocorrelation buffer).
4. Predict on the ``k`` test groups.

Each group appears in ``C(N - 1, k - 1)`` combinations. By symmetric pairing
across combinations one can construct exactly ``C(N - 1, k - 1)`` synthetic
out-of-sample paths, each containing one prediction per group covering the
entire dataset.

The number of paths is **independent of ``k``** for fixed ``N``: ``C(5, 1) =
C(5, 4) = 5`` for ``N = 6``. The Blueprint §8.1 prescription uses ``k`` to
control the resampling intensity (more paths means stronger PBO power but
more compute).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
import numpy.typing as npt

DEFAULT_N_GROUPS: int = 6
DEFAULT_N_TEST_GROUPS: int = 2
DEFAULT_EMBARGO_PCT: float = 0.01


@dataclass(frozen=True, slots=True)
class CPCVFold:
    """One element of the ``C(N, k)`` enumeration.

    Attributes
    ----------
    combination_index
        Position of this fold in the ``C(N, k)`` enumeration (0-indexed).
    train_idx, test_idx
        Sample indices into the *original* (pre-sort) row order. Disjoint.
    test_groups
        Tuple of group indices (in ``[0, N)``) that make up this fold's
        test set.
    """

    combination_index: int
    train_idx: npt.NDArray[np.int64]
    test_idx: npt.NDArray[np.int64]
    test_groups: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CombinatoriallyPurgedKFold:
    """CPCV splitter — generates ``C(N, k)`` purged + embargoed folds.

    Parameters
    ----------
    n_groups
        Total number of contiguous groups to split the timeline into.
    n_test_groups
        Number of groups in each test combination. Must be ``< n_groups``.
    embargo_pct
        Embargo width as a fraction of the total sample count, applied
        after each contiguous test block.
    """

    n_groups: int = DEFAULT_N_GROUPS
    n_test_groups: int = DEFAULT_N_TEST_GROUPS
    embargo_pct: float = DEFAULT_EMBARGO_PCT

    def __post_init__(self) -> None:
        if self.n_groups < 2:
            raise ValueError(f"n_groups must be ≥ 2, got {self.n_groups}")
        if self.n_test_groups < 1 or self.n_test_groups >= self.n_groups:
            raise ValueError(
                f"need 1 ≤ n_test_groups < n_groups, got {self.n_test_groups} / {self.n_groups}"
            )
        if not 0.0 <= self.embargo_pct < 0.5:
            raise ValueError(f"embargo_pct must be in [0, 0.5), got {self.embargo_pct}")

    def get_n_splits(self) -> int:
        """Return ``C(n_groups, n_test_groups)``."""
        return comb(self.n_groups, self.n_test_groups)

    def get_n_paths(self) -> int:
        """Return the number of synthetic OOS paths = ``C(n_groups-1, n_test_groups-1)``."""
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    def split(
        self,
        t0: npt.NDArray[np.int64] | npt.NDArray[np.floating],
        t1: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    ) -> Iterator[CPCVFold]:
        """Yield ``C(n_groups, n_test_groups)`` purged + embargoed folds.

        Indices reference the *original* (pre-sort) row positions; the
        splitter sorts internally by ``t0``.
        """
        t0_arr = np.asarray(t0)
        t1_arr = np.asarray(t1)
        if t0_arr.shape != t1_arr.shape or t0_arr.ndim != 1:
            raise ValueError(
                f"t0 and t1 must be 1-D and same shape; got {t0_arr.shape} / {t1_arr.shape}"
            )
        n = t0_arr.size
        if n < self.n_groups:
            raise ValueError(f"need ≥ n_groups samples; got {n} vs {self.n_groups}")
        if np.any(t1_arr < t0_arr):
            raise ValueError("every t1[i] must be ≥ t0[i]")

        sort_order = np.argsort(t0_arr, kind="mergesort")
        t0_sorted = t0_arr[sort_order]
        t1_sorted = t1_arr[sort_order]

        # Partition the sorted index into n_groups contiguous chunks.
        group_edges = np.linspace(0, n, self.n_groups + 1, dtype=np.int64)
        groups: list[npt.NDArray[np.int64]] = [
            np.arange(group_edges[g], group_edges[g + 1], dtype=np.int64)
            for g in range(self.n_groups)
        ]

        embargo_size = int(np.floor(n * self.embargo_pct))

        for idx, test_group_tuple in enumerate(
            combinations(range(self.n_groups), self.n_test_groups)
        ):
            # Build sorted test-index set.
            test_idx_sorted = np.concatenate([groups[g] for g in test_group_tuple]).astype(np.int64)
            # Find contiguous blocks of test groups so the embargo can be
            # applied after each block (not after every group, which would
            # over-purge when test groups are adjacent).
            test_block_ranges = _contiguous_test_blocks(test_group_tuple, group_edges)

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx_sorted] = False

            # PURGE: drop any train sample whose label horizon overlaps any
            # test sample's horizon. The cheapest correct check is on the
            # *union* of test horizons — purge if t0 ≤ max(test_t1) AND
            # t1 ≥ min(test_t0) for each contiguous test block.
            for block_start, block_stop in test_block_ranges:
                test_block_t0_min = t0_sorted[block_start]
                test_block_t1_max = t1_sorted[block_start:block_stop].max()
                overlap = (t0_sorted <= test_block_t1_max) & (t1_sorted >= test_block_t0_min)
                train_mask &= ~overlap

                # EMBARGO: zero out the `embargo_size` sorted positions
                # immediately after the block end.
                if embargo_size > 0:
                    embargo_stop = min(n, block_stop + embargo_size)
                    train_mask[block_stop:embargo_stop] = False

            train_idx_sorted = np.where(train_mask)[0]
            yield CPCVFold(
                combination_index=idx,
                train_idx=sort_order[train_idx_sorted].astype(np.int64),
                test_idx=sort_order[test_idx_sorted].astype(np.int64),
                test_groups=test_group_tuple,
            )


def _contiguous_test_blocks(
    test_groups: tuple[int, ...],
    group_edges: npt.NDArray[np.int64],
) -> list[tuple[int, int]]:
    """Identify contiguous runs of test groups and return their sorted-index spans."""
    if not test_groups:
        return []
    sorted_groups = sorted(test_groups)
    blocks: list[tuple[int, int]] = []
    block_start = sorted_groups[0]
    block_end = sorted_groups[0]
    for g in sorted_groups[1:]:
        if g == block_end + 1:
            block_end = g
        else:
            blocks.append((int(group_edges[block_start]), int(group_edges[block_end + 1])))
            block_start = g
            block_end = g
    blocks.append((int(group_edges[block_start]), int(group_edges[block_end + 1])))
    return blocks


def construct_oos_paths(
    fold_predictions: dict[tuple[int, ...], dict[int, npt.NDArray[np.floating]]],
    n_groups: int,
    n_test_groups: int,
    *,
    n_samples_per_group: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.float64]:
    """Assemble synthetic OOS paths from per-combination predictions.

    Each combination ``c`` contains predictions for ``n_test_groups`` groups.
    We need to weave them into ``C(n_groups - 1, n_test_groups - 1)`` paths
    where each path covers all ``n_groups`` groups exactly once.

    Parameters
    ----------
    fold_predictions
        ``{test_group_tuple: {group_index: predictions_array}}``. The keys at
        the first level are the tuples yielded by
        :meth:`CombinatoriallyPurgedKFold.split` as ``test_groups``. The
        inner dict maps each group in that tuple to the per-row prediction
        array for that group's slice of the dataset.
    n_groups, n_test_groups
        CPCV parameters.
    n_samples_per_group
        Optional ``(n_groups,)`` array giving the row count of each group.
        When ``None``, inferred from ``fold_predictions``.

    Returns
    -------
    Array of shape ``(n_paths, n_total_samples)`` where ``n_paths =
    C(n_groups - 1, n_test_groups - 1)``. Each row is one complete synthetic
    OOS prediction path over the entire dataset, sorted by group order.

    Notes
    -----
    The path-construction algorithm follows López de Prado 2018 §12.4.4 — a
    bipartite-matching round-robin that guarantees:

    1. Every combination contributes to *exactly* ``n_test_groups`` paths.
    2. Every group is covered exactly once per path.
    3. The same combination is never reused for the same group across paths.
    """
    combinations_list = list(fold_predictions.keys())
    n_combinations = len(combinations_list)
    expected = comb(n_groups, n_test_groups)
    if n_combinations != expected:
        raise ValueError(f"expected {expected} combinations, got {n_combinations}")
    n_paths = comb(n_groups - 1, n_test_groups - 1)

    # Build a matrix M[combination, group] = predictions slot reference.
    # Walk combinations and, for each, pick a free path slot for each of its
    # test groups. Each combination must end up in exactly n_test_groups
    # paths; each group must end up in n_paths slots (one per path).
    if n_samples_per_group is None:
        sizes_per_group = np.zeros(n_groups, dtype=np.int64)
        for preds_by_group in fold_predictions.values():
            for g, preds in preds_by_group.items():
                if sizes_per_group[g] != 0 and sizes_per_group[g] != preds.shape[0]:
                    raise ValueError(
                        f"group {g} has inconsistent sizes across combinations: "
                        f"{sizes_per_group[g]} vs {preds.shape[0]}"
                    )
                sizes_per_group[g] = preds.shape[0]
        n_samples_per_group = sizes_per_group

    total_samples = int(n_samples_per_group.sum())
    group_offsets = np.concatenate(([0], np.cumsum(n_samples_per_group))).astype(np.int64)

    # For each group, list of combinations that test that group.
    combos_per_group: dict[int, list[tuple[int, ...]]] = {g: [] for g in range(n_groups)}
    for combo in combinations_list:
        for g in combo:
            combos_per_group[g].append(combo)
    # Each group must show up in exactly C(N-1, k-1) combinations.
    for g, combos in combos_per_group.items():
        if len(combos) != n_paths:
            raise ValueError(f"group {g} appears in {len(combos)} combinations, expected {n_paths}")

    # Round-robin: for path p, iterate groups in order and pop the p-th
    # combination from combos_per_group[g].
    paths = np.full((n_paths, total_samples), np.nan, dtype=np.float64)
    for path_idx in range(n_paths):
        for g in range(n_groups):
            combo = combos_per_group[g][path_idx]
            preds = fold_predictions[combo][g]
            start = group_offsets[g]
            stop = group_offsets[g + 1]
            paths[path_idx, start:stop] = preds

    return paths
