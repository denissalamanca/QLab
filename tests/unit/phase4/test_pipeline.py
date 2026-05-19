"""Phase 4 — pipeline orchestrator + Blueprint §6.3 DoD.

Blueprint §6.3 unit tests verbatim:

* **Dimensionality Reduction Check:** ``len(selected_clusters) < len(original_features)``.
* **Orthogonality Check:** Post-clustering, inter-cluster correlation must be
  mathematically minimized (i.e., inter-cluster |ρ| < intra-cluster |ρ|).

Additional invariants enforced here:

* The SFI fallback engages when Clustered MDA fails to reduce the surviving
  feature count by ≥ 20 % (CLAUDE.md anti-shortcut: every selection step must
  produce meaningful reduction).
* Banned-method import audit on the entire ``selection`` package (no
  ``KFold``, ``TimeSeriesSplit``, ``MDI``, ``Gini``, ``feature_importances_``,
  ``accuracy_score``, ``roc_auc_score`` used as primary).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from afml.selection import mda as mda_module
from afml.selection.pipeline import select_features
from tests.unit.phase4.conftest import SyntheticDataset


@pytest.mark.phase4
def test_pipeline_reduces_dimensionality(synthetic_classification: SyntheticDataset) -> None:
    """Blueprint §6.3: ``len(selected) < len(original)``."""
    ds = synthetic_classification
    result = select_features(ds.X, ds.y, ds.t0, ds.t1, n_splits=4, random_state=0)
    assert len(result.surviving_features) < len(ds.feature_columns)


@pytest.mark.phase4
def test_pipeline_keeps_signal_drops_noise(
    synthetic_classification: SyntheticDataset,
) -> None:
    """The 3 signal features must survive; the 9 noise features must not."""
    ds = synthetic_classification
    result = select_features(ds.X, ds.y, ds.t0, ds.t1, n_splits=4, random_state=0)
    for sig in ds.signal_cols:
        assert sig in result.surviving_features, f"signal feature {sig} dropped"
    for noise in ds.correlated_noise_cols + ds.independent_noise_cols:
        assert noise not in result.surviving_features, f"noise feature {noise} kept"


@pytest.mark.phase4
def test_pipeline_post_clustering_orthogonality(
    synthetic_classification: SyntheticDataset,
) -> None:
    """Blueprint §6.3: inter-cluster correlation must be mathematically minimized.

    Concretely: averaging |ρ| across kept-cluster-vs-kept-cluster pairs must
    be materially lower than the average intra-cluster |ρ|.
    """
    ds = synthetic_classification
    result = select_features(ds.X, ds.y, ds.t0, ds.t1, n_splits=4, random_state=0)

    X_full = ds.X.to_numpy().astype(np.float64)
    rho = np.corrcoef(X_full, rowvar=False)
    n_features = X_full.shape[1]
    labels = result.onc.labels

    intra: list[float] = []
    inter: list[float] = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if labels[i] == labels[j]:
                intra.append(abs(rho[i, j]))
            else:
                inter.append(abs(rho[i, j]))
    assert intra and inter
    mean_intra = float(np.mean(intra))
    mean_inter = float(np.mean(inter))
    assert mean_inter < mean_intra - 0.2, (
        f"orthogonality not achieved: mean intra |rho|={mean_intra:.3f}, "
        f"mean inter |rho|={mean_inter:.3f}"
    )


@pytest.mark.phase4
def test_pipeline_all_noise_drops_everything(
    synthetic_all_noise: SyntheticDataset,
) -> None:
    """On pure noise, Clustered MDA correctly rejects every cluster; reduction
    is 100 %; SFI is NOT engaged (gate satisfied trivially)."""
    ds = synthetic_all_noise
    result = select_features(
        ds.X, ds.y, ds.t0, ds.t1, n_splits=4, random_state=0, min_reduction_pct=0.20
    )
    assert not result.used_sfi_fallback
    assert result.surviving_features == []
    assert result.mda.kept_clusters == []


@pytest.mark.phase4
def test_pipeline_triggers_sfi_fallback_when_mda_keeps_everything(
    synthetic_redundant_signal: SyntheticDataset,
) -> None:
    """If every feature is a noisy copy of the same signal, MDA's one
    informative cluster contains everything — 0 % reduction — and the
    pipeline falls back to SFI to enforce the ≥ 20 % reduction floor."""
    ds = synthetic_redundant_signal
    result = select_features(
        ds.X, ds.y, ds.t0, ds.t1, n_splits=4, random_state=0, min_reduction_pct=0.20
    )
    assert result.used_sfi_fallback
    assert result.sfi is not None
    # Even via SFI, the surviving set must respect the reduction floor.
    assert len(result.surviving_features) <= int(np.floor(len(ds.feature_columns) * (1.0 - 0.20)))
    # SFI must keep at least one feature — the strongest signal carrier.
    assert len(result.surviving_features) >= 1


@pytest.mark.phase4
def test_pipeline_result_carries_all_diagnostics(
    synthetic_classification: SyntheticDataset,
) -> None:
    ds = synthetic_classification
    result = select_features(ds.X, ds.y, ds.t0, ds.t1, n_splits=4, random_state=0)
    assert result.feature_names == ds.feature_columns
    assert result.distance_matrix.shape == (len(ds.feature_columns), len(ds.feature_columns))
    # The cluster_to_features map covers every feature.
    flat = [c for cols in result.cluster_to_features.values() for c in cols]
    assert sorted(flat) == sorted(ds.feature_columns)
    # ONC silhouette curve covers the full sweep.
    expected_keys = set(range(2, len(ds.feature_columns)))
    assert set(result.onc.silhouette_curve.keys()) == expected_keys


@pytest.mark.phase4
def test_pipeline_deterministic_with_same_seed(
    synthetic_classification: SyntheticDataset,
) -> None:
    ds = synthetic_classification
    r1 = select_features(ds.X, ds.y, ds.t0, ds.t1, random_state=42)
    r2 = select_features(ds.X, ds.y, ds.t0, ds.t1, random_state=42)
    assert r1.surviving_features == r2.surviving_features
    np.testing.assert_array_equal(r1.onc.labels, r2.onc.labels)


@pytest.mark.phase4
def test_pipeline_rejects_too_few_features(
    synthetic_classification: SyntheticDataset,
) -> None:
    ds = synthetic_classification
    with pytest.raises(ValueError, match="≥ 2"):
        select_features(ds.X.select(ds.feature_columns[:1]), ds.y, ds.t0, ds.t1, random_state=0)


@pytest.mark.phase4
def test_selection_module_does_not_use_banned_methods() -> None:
    """Static guard — CLAUDE.md anti-shortcut rules.

    Parses each ``afml/selection`` source file as Python AST and asserts:

    1. No ``import`` or ``from … import`` brings in ``sklearn.model_selection``
       members (``KFold``, ``TimeSeriesSplit``).
    2. No attribute access uses MDI / Gini's ``feature_importances_`` /
       ``feature_importance_``.
    3. ``sklearn.metrics.accuracy_score`` / ``roc_auc_score`` are not called as
       the primary scoring function (we use ``log_loss`` exclusively).

    Docstring / comment prose mentioning the banned symbols is permitted — the
    AST walk only looks at executable nodes.
    """
    selection_dir = Path("src/afml/selection")
    banned_from_imports = {
        ("sklearn.model_selection", "KFold"),
        ("sklearn.model_selection", "TimeSeriesSplit"),
        ("sklearn.metrics", "accuracy_score"),
        ("sklearn.metrics", "roc_auc_score"),
    }
    banned_attributes = {"feature_importances_", "feature_importance_"}

    for py in selection_dir.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    assert (module, alias.name) not in banned_from_imports, (
                        f"banned import {module}.{alias.name} in {py}"
                    )
            elif isinstance(node, ast.Attribute):
                assert node.attr not in banned_attributes, (
                    f"banned attribute access {node.attr!r} in {py} at line {node.lineno}"
                )


@pytest.mark.phase4
def test_purged_kfold_imported_not_kfold() -> None:
    """Defensive sanity: ``select_features`` actually uses ``PurgedKFold``."""
    src = inspect.getsource(mda_module)
    assert "PurgedKFold" in src
    assert "from sklearn.model_selection import KFold" not in src
