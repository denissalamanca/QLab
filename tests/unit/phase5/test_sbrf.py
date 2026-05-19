"""Phase 5 — Sequentially Bootstrapped Random Forest."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
from sklearn.base import is_classifier

from afml.modeling import (
    SequentiallyBootstrappedRandomForest,
    indicator_matrix,
)
from afml.modeling import sbrf as sbrf_module


@pytest.mark.phase5
def test_sbrf_is_a_classifier() -> None:
    """sklearn helpers must recognise the SBRF as a classifier — otherwise
    ``CalibratedClassifierCV`` and ``FrozenEstimator`` cannot wrap it."""
    sbrf = SequentiallyBootstrappedRandomForest()
    assert is_classifier(sbrf)
    assert sbrf.__sklearn_tags__().estimator_type == "classifier"


@pytest.mark.phase5
def test_sbrf_fits_and_predicts() -> None:
    rng = np.random.default_rng(0)
    n = 400
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] > 0.0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 3
    ind = indicator_matrix(t0, t1)

    sbrf = SequentiallyBootstrappedRandomForest(n_estimators=20, random_state=0)
    sbrf.fit(X, y, ind)
    proba = sbrf.predict_proba(X[:50])
    assert proba.shape == (50, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.phase5
def test_sbrf_n_estimators_matches_attribute() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] > 0.0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    ind = indicator_matrix(t0, t1)

    sbrf = SequentiallyBootstrappedRandomForest(n_estimators=15, random_state=0)
    sbrf.fit(X, y, ind)
    assert len(sbrf.estimators_) == 15
    expected_classes = 2
    expected_features = 3
    assert sbrf.n_classes_ == expected_classes  # binary meta-label
    assert sbrf.n_features_in_ == expected_features


@pytest.mark.phase5
def test_sbrf_reproducible_with_seed() -> None:
    rng = np.random.default_rng(0)
    n = 300
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] > 0.0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    ind = indicator_matrix(t0, t1)

    sbrf1 = SequentiallyBootstrappedRandomForest(n_estimators=20, random_state=42)
    sbrf1.fit(X, y, ind)
    sbrf2 = SequentiallyBootstrappedRandomForest(n_estimators=20, random_state=42)
    sbrf2.fit(X, y, ind)
    np.testing.assert_allclose(sbrf1.predict_proba(X[:50]), sbrf2.predict_proba(X[:50]))


@pytest.mark.phase5
def test_sbrf_rejects_shape_mismatch() -> None:
    rng = np.random.default_rng(0)
    n = 100
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] > 0.0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 1
    ind = indicator_matrix(t0, t1)
    sbrf = SequentiallyBootstrappedRandomForest(n_estimators=5, random_state=0)
    with pytest.raises(ValueError, match="y shape"):
        sbrf.fit(X, y[:90], ind)
    with pytest.raises(ValueError, match="indicator_mat"):
        sbrf.fit(X, y, ind[:, :90])


@pytest.mark.phase5
def test_sbrf_uses_sequential_bootstrap_not_uniform() -> None:
    """The internal bootstrap path must call our :func:`sequential_bootstrap`
    rather than sklearn's default uniform bootstrap. AST-level check."""
    src = inspect.getsource(sbrf_module)
    tree = ast.parse(src)
    bootstrap_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "sequential_bootstrap"
    ]
    assert len(bootstrap_calls) >= 1, "SBRF must call sequential_bootstrap"
