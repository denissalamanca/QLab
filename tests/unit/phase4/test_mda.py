"""Phase 4 — Clustered Mean Decrease Accuracy (NLL-based)."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from afml.selection import mda as mda_module
from afml.selection.clustering import cluster_features_onc
from afml.selection.distance import afml_distance_matrix
from afml.selection.mda import clustered_mda
from tests.unit.phase4.conftest import SyntheticDataset


@pytest.mark.phase4
def test_clustered_mda_identifies_signal_cluster(
    synthetic_classification: SyntheticDataset,
) -> None:
    """The cluster containing the signal features must have the highest
    importance and a significant p-value."""
    ds = synthetic_classification
    X = ds.X.to_numpy().astype(np.float64)
    y = ds.y
    D = afml_distance_matrix(X)
    onc = cluster_features_onc(D)
    result = clustered_mda(X, y, onc.labels, ds.t0, ds.t1, n_splits=4, random_state=0)

    # Find which cluster holds the signal columns.
    signal_col_indices = [ds.feature_columns.index(c) for c in ds.signal_cols]
    signal_cluster_ids = {int(onc.labels[i]) for i in signal_col_indices}
    assert len(signal_cluster_ids) == 1, (
        f"signal features did not cluster together: {signal_cluster_ids}"
    )
    signal_cluster_id = signal_cluster_ids.pop()

    # The signal cluster must be in the kept set.
    assert signal_cluster_id in result.kept_clusters
    # Its importance dominates every other cluster.
    sig_imp = result.cluster_importance[signal_cluster_id]
    for other in result.cluster_importance:
        if other == signal_cluster_id:
            continue
        assert sig_imp > result.cluster_importance[other], (
            f"signal cluster importance {sig_imp:.4f} not greater than "
            f"cluster {other}: {result.cluster_importance[other]:.4f}"
        )
    # Its p-value clears the significance bar comfortably.
    assert result.cluster_pvalue[signal_cluster_id] < 0.05


@pytest.mark.phase4
def test_clustered_mda_rejects_pure_noise_clusters(
    synthetic_classification: SyntheticDataset,
) -> None:
    """No noise cluster (correlated or independent) should pass significance."""
    ds = synthetic_classification
    X = ds.X.to_numpy().astype(np.float64)
    y = ds.y
    D = afml_distance_matrix(X)
    onc = cluster_features_onc(D)
    result = clustered_mda(X, y, onc.labels, ds.t0, ds.t1, n_splits=4, random_state=0)

    noise_col_indices = [
        ds.feature_columns.index(c) for c in ds.correlated_noise_cols + ds.independent_noise_cols
    ]
    noise_cluster_ids = {int(onc.labels[i]) for i in noise_col_indices}
    for noise_id in noise_cluster_ids:
        # Either the cluster's importance is ≤ 0, or its p-value misses the bar.
        if result.cluster_importance[noise_id] > 0:
            assert result.cluster_pvalue[noise_id] > 0.05, (
                f"noise cluster {noise_id} falsely accepted: "
                f"imp={result.cluster_importance[noise_id]:.4f}, "
                f"p={result.cluster_pvalue[noise_id]:.4f}"
            )


@pytest.mark.phase4
def test_clustered_mda_fold_importances_shape(
    synthetic_classification: SyntheticDataset,
) -> None:
    ds = synthetic_classification
    X = ds.X.to_numpy().astype(np.float64)
    y = ds.y
    D = afml_distance_matrix(X)
    onc = cluster_features_onc(D)
    result = clustered_mda(X, y, onc.labels, ds.t0, ds.t1, n_splits=5, random_state=0)
    assert result.fold_importances.shape == (5, onc.n_clusters)


@pytest.mark.phase4
def test_clustered_mda_reproducible_with_same_seed(
    synthetic_classification: SyntheticDataset,
) -> None:
    ds = synthetic_classification
    X = ds.X.to_numpy().astype(np.float64)
    y = ds.y
    D = afml_distance_matrix(X)
    onc = cluster_features_onc(D)
    r1 = clustered_mda(X, y, onc.labels, ds.t0, ds.t1, random_state=42)
    r2 = clustered_mda(X, y, onc.labels, ds.t0, ds.t1, random_state=42)
    np.testing.assert_allclose(r1.fold_importances, r2.fold_importances)
    assert r1.kept_clusters == r2.kept_clusters


@pytest.mark.phase4
def test_clustered_mda_rejects_bad_input() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    y = rng.integers(0, 2, size=100)
    labels = np.zeros(4, dtype=np.int64)
    t0 = np.arange(100, dtype=np.int64)
    t1 = t0 + 1
    # Non-binary y
    bad_y = rng.integers(0, 3, size=100)
    with pytest.raises(ValueError, match="binary"):
        clustered_mda(X, bad_y, labels, t0, t1)
    # NaN in X
    bad_X = X.copy()
    bad_X[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        clustered_mda(bad_X, y, labels, t0, t1)
    # Wrong y shape
    with pytest.raises(ValueError, match="y shape"):
        clustered_mda(X, y[:90], labels, t0, t1)
    # Wrong cluster_labels shape
    with pytest.raises(ValueError, match="cluster_labels"):
        clustered_mda(X, y, np.zeros(3, dtype=np.int64), t0, t1)


@pytest.mark.phase4
def test_clustered_mda_does_not_call_feature_importances() -> None:
    """Anti-shortcut: the MDA implementation must NOT call sklearn's MDI / Gini
    ``feature_importances_`` attribute.

    AST-level check — docstring prose mentioning the banned name is OK; only
    actual attribute access is forbidden.
    """
    tree = ast.parse(inspect.getsource(mda_module))
    banned_attributes = {"feature_importances_", "feature_importance_"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in banned_attributes, (
                f"banned attribute access {node.attr!r} at mda.py:{node.lineno}"
            )
