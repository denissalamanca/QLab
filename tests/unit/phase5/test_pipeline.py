"""Phase 5 — Brain 2 orchestrator + Blueprint §7.3 DoD.

Blueprint §7.3 DoD (verbatim):

* **Brier Score Optimization:**
  ``brier_score_loss(y_true, y_prob) < brier_score_loss(y_true, naive_baseline_prob)``
* **Purging / Embargo Verification:**
  for every fold, ``max(train_times[train_idx]) + embargo_time < min(test_times[test_idx])``.

Both assertions must hold on every walk-forward fold the orchestrator produces.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from afml.modeling import train_brain_two
from tests.unit.phase5.conftest import MetaLabelDataset


@pytest.mark.phase5
def test_brain_two_passes_blueprint_dod(
    predictable_meta_labels: MetaLabelDataset,
) -> None:
    """The aggregate DoD: every fold beats the naive baseline AND passes the
    index-intersection check."""
    ds = predictable_meta_labels
    result = train_brain_two(
        ds.X,
        ds.y,
        ds.t0,
        ds.t1,
        n_splits=4,
        embargo_pct=0.02,
        n_estimators=50,
        random_state=0,
    )
    assert result.passes_phase5_dod
    min_folds = 4
    assert len(result.fold_diagnostics) >= min_folds


@pytest.mark.phase5
def test_brain_two_brier_optimization_per_fold(
    predictable_meta_labels: MetaLabelDataset,
) -> None:
    """Blueprint §7.3.1 — calibrated Brier strictly below naive baseline."""
    ds = predictable_meta_labels
    result = train_brain_two(ds.X, ds.y, ds.t0, ds.t1, n_splits=4, n_estimators=50, random_state=0)
    for fd in result.fold_diagnostics:
        assert fd.beats_naive_baseline, (
            f"fold {fd.fold_index}: brier_calibrated={fd.brier_calibrated:.4f} "
            f">= brier_naive={fd.brier_naive_baseline:.4f}"
        )


@pytest.mark.phase5
def test_brain_two_index_intersection_per_fold(
    predictable_meta_labels: MetaLabelDataset,
) -> None:
    """Blueprint §7.3.2 — index intersection on the literal ``t0`` / ``t1``."""
    ds = predictable_meta_labels
    result = train_brain_two(ds.X, ds.y, ds.t0, ds.t1, n_splits=4, n_estimators=50, random_state=0)
    for fd in result.fold_diagnostics:
        assert fd.passes_index_intersection, (
            f"fold {fd.fold_index}: train_t1_max={fd.train_t1_max}, "
            f"embargo={fd.embargo_size}, holdout_t0_min={fd.holdout_t0_min}"
        )


@pytest.mark.phase5
def test_brain_two_returns_usable_calibrated_classifier(
    predictable_meta_labels: MetaLabelDataset,
) -> None:
    """``last_calibration.calibrated`` must be ready to call ``predict_proba``
    on fresh data — that's the artefact Phase 7 will deploy."""
    ds = predictable_meta_labels
    result = train_brain_two(ds.X, ds.y, ds.t0, ds.t1, n_splits=3, n_estimators=30, random_state=0)
    proba = result.last_calibration.calibrated.predict_proba(ds.X[:50])
    assert proba.shape == (50, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    # Probabilities must lie in [0, 1].
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


@pytest.mark.phase5
def test_brain_two_average_uniqueness_is_sane(
    predictable_meta_labels: MetaLabelDataset,
) -> None:
    """ū_i should be in (0, 1] and roughly uniform on this synthetic data
    (every event has the same horizon length so overlap structure is uniform)."""
    ds = predictable_meta_labels
    result = train_brain_two(ds.X, ds.y, ds.t0, ds.t1, n_splits=3, n_estimators=30, random_state=0)
    avg_u = result.average_uniqueness_per_event
    assert avg_u.shape == (ds.X.shape[0],)
    assert avg_u.min() > 0.0
    assert avg_u.max() <= 1.0


@pytest.mark.phase5
def test_brain_two_reproducible_with_same_seed(
    predictable_meta_labels: MetaLabelDataset,
) -> None:
    ds = predictable_meta_labels
    r1 = train_brain_two(ds.X, ds.y, ds.t0, ds.t1, n_splits=3, n_estimators=30, random_state=42)
    r2 = train_brain_two(ds.X, ds.y, ds.t0, ds.t1, n_splits=3, n_estimators=30, random_state=42)
    # Per-fold Briers must match exactly across the two runs.
    b1 = [f.brier_calibrated for f in r1.fold_diagnostics]
    b2 = [f.brier_calibrated for f in r2.fold_diagnostics]
    np.testing.assert_allclose(b1, b2)


@pytest.mark.phase5
def test_modeling_module_does_not_use_banned_methods() -> None:
    """Static AST guard — no ``KFold`` / ``TimeSeriesSplit`` / ``accuracy_score``
    / ``roc_auc_score`` / ``feature_importances_`` anywhere in
    ``src/afml/modeling/``.

    Calibration accesses ``predict_proba`` only; SBRF inherits ``fit`` /
    ``predict`` but never reads tree-level Gini importances.
    """
    modeling_dir = Path("src/afml/modeling")
    banned_from_imports = {
        ("sklearn.model_selection", "KFold"),
        ("sklearn.model_selection", "TimeSeriesSplit"),
        ("sklearn.metrics", "accuracy_score"),
        ("sklearn.metrics", "roc_auc_score"),
    }
    banned_attributes = {"feature_importances_", "feature_importance_"}
    for py in modeling_dir.rglob("*.py"):
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
                    f"banned attribute access {node.attr!r} in {py}:{node.lineno}"
                )
