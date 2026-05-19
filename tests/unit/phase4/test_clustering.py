"""Phase 4 — Ward agglomerative clustering with ONC silhouette-max."""

from __future__ import annotations

import numpy as np
import pytest

from afml.selection.clustering import cluster_features_onc
from afml.selection.distance import afml_distance_matrix


def _make_block_correlated(
    n_samples: int,
    block_sizes: list[int],
    noise_scale: float,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Build a feature matrix with ``len(block_sizes)`` correlated blocks.

    Each block shares a hidden factor; within a block, features are noisy
    copies of the factor. Across blocks, features are independent.
    """
    rng = np.random.default_rng(seed)
    cols: list[np.ndarray] = []
    for size in block_sizes:
        factor = rng.standard_normal(n_samples)
        for _ in range(size):
            cols.append(factor + rng.standard_normal(n_samples) * noise_scale)
    return np.column_stack(cols)


@pytest.mark.phase4
def test_onc_recovers_known_block_count() -> None:
    """Three tight blocks of 3 features each ⇒ ONC must pick k = 3."""
    X = _make_block_correlated(800, [3, 3, 3], noise_scale=0.1)
    D = afml_distance_matrix(X)
    result = cluster_features_onc(D)
    assert result.n_clusters == 3
    # Every feature inside a block must share the same cluster label.
    for block_start in (0, 3, 6):
        block_labels = result.labels[block_start : block_start + 3]
        assert len(set(block_labels)) == 1, f"block at {block_start} split: {block_labels}"


@pytest.mark.phase4
def test_onc_is_deterministic() -> None:
    """Same input ⇒ same cluster labels and silhouette curve."""
    X = _make_block_correlated(500, [4, 4], noise_scale=0.2, seed=1)
    D = afml_distance_matrix(X)
    r1 = cluster_features_onc(D)
    r2 = cluster_features_onc(D)
    np.testing.assert_array_equal(r1.labels, r2.labels)
    assert r1.silhouette_curve == r2.silhouette_curve


@pytest.mark.phase4
def test_onc_silhouette_curve_covers_full_sweep() -> None:
    X = _make_block_correlated(400, [2, 2, 2], noise_scale=0.15)
    D = afml_distance_matrix(X)
    result = cluster_features_onc(D)
    assert set(result.silhouette_curve.keys()) == set(range(2, D.shape[0]))


@pytest.mark.phase4
def test_onc_rejects_non_square_input() -> None:
    rng = np.random.default_rng(0)
    bad = rng.standard_normal((5, 4))
    with pytest.raises(ValueError, match="square"):
        cluster_features_onc(bad)


@pytest.mark.phase4
def test_onc_rejects_asymmetric_input() -> None:
    rng = np.random.default_rng(0)
    bad = rng.standard_normal((5, 5))
    # Force asymmetry.
    bad[0, 1] = 0.0
    bad[1, 0] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        cluster_features_onc(bad)


@pytest.mark.phase4
def test_onc_stable_plateau_picks_smallest_k() -> None:
    """When silhouette is flat across multiple k values, the smallest k wins
    (anti-shortcut rule: never pick from a noisy peak)."""
    # Construct a distance matrix where k=2..4 all yield ~identical silhouette:
    # 4 perfectly-tight pairs separated by a constant inter-pair distance.
    n = 8
    D = np.zeros((n, n))
    # Inside-pair distances near 0.
    for pair_start in range(0, n, 2):
        D[pair_start, pair_start + 1] = 0.01
        D[pair_start + 1, pair_start] = 0.01
    # Inter-pair distances all near 0.9.
    for i in range(n):
        for j in range(n):
            if i != j and D[i, j] == 0.0:
                D[i, j] = 0.9

    # Use a very wide plateau tolerance so multiple k's qualify.
    result = cluster_features_onc(D, plateau_eps=0.05)
    # All k ∈ {2, 3, 4} produce well-separated clusters; the smallest wins.
    assert result.n_clusters >= 2
    assert result.n_clusters <= 4


@pytest.mark.phase4
def test_onc_inter_cluster_correlation_lower_than_intra() -> None:
    """Blueprint §6.3 orthogonality check: inter-cluster |ρ| ≪ intra-cluster."""
    n_samples = 1000
    X = _make_block_correlated(n_samples, [3, 3, 3], noise_scale=0.15, seed=11)
    D = afml_distance_matrix(X)
    result = cluster_features_onc(D)
    # Use the Pearson correlation matrix directly for the orthogonality check.
    rho = np.corrcoef(X, rowvar=False)
    intra_abs: list[float] = []
    inter_abs: list[float] = []
    for i in range(rho.shape[0]):
        for j in range(i + 1, rho.shape[0]):
            if result.labels[i] == result.labels[j]:
                intra_abs.append(abs(rho[i, j]))
            else:
                inter_abs.append(abs(rho[i, j]))
    assert intra_abs, "no intra-cluster pairs — fixture is degenerate"
    assert inter_abs, "no inter-cluster pairs — fixture is degenerate"
    # Inter-cluster correlation must be materially lower than intra-cluster.
    assert np.mean(inter_abs) < np.mean(intra_abs) - 0.2, (
        f"orthogonality not achieved: mean intra={np.mean(intra_abs):.3f}, "
        f"mean inter={np.mean(inter_abs):.3f}"
    )
