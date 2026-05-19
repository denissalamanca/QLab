"""Phase 8 — SHAP feature-importance concept-drift detection (Blueprint §10.2)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from afml.monitoring.shap_drift import (
    compute_shap_importance,
    detect_concept_drift,
    spearman_rank_correlation,
)


@pytest.mark.phase8
def test_identical_importance_no_drift() -> None:
    imp = np.array([0.5, 0.3, 0.1, 0.07, 0.03])
    result = detect_concept_drift(imp, imp.copy())
    assert result.spearman_rank_corr == pytest.approx(1.0)
    assert result.drifted is False


@pytest.mark.phase8
def test_reversed_importance_triggers_drift() -> None:
    """Blueprint §10.2 — a reversed importance ranking gives strongly
    negative rank correlation (< 0.5) ⇒ CONCEPT_DRIFT_ALERT."""
    train = np.array([0.5, 0.3, 0.1, 0.07, 0.03])
    live = train[::-1].copy()
    result = detect_concept_drift(train, live)
    assert result.spearman_rank_corr < 0.5
    assert result.drifted is True


@pytest.mark.phase8
def test_mild_reshuffle_below_threshold_triggers() -> None:
    """A ranking that has decorrelated past 0.5 must alert."""
    train = np.array([0.40, 0.30, 0.15, 0.10, 0.05])
    # Shuffle so the rank correlation lands below 0.5.
    live = np.array([0.05, 0.40, 0.10, 0.30, 0.15])
    result = detect_concept_drift(train, live)
    if result.spearman_rank_corr < 0.5:
        assert result.drifted is True
    else:
        assert result.drifted is False


@pytest.mark.phase8
def test_preserved_ranking_no_drift() -> None:
    """Same ranking, different magnitudes (vol scaling) ⇒ rho=1, no drift."""
    train = np.array([0.5, 0.3, 0.1, 0.07, 0.03])
    live = train * 3.0  # magnitudes change, order preserved
    result = detect_concept_drift(train, live)
    assert result.spearman_rank_corr == pytest.approx(1.0)
    assert result.drifted is False


@pytest.mark.phase8
def test_spearman_constant_vector_returns_zero() -> None:
    a = np.array([1.0, 2.0, 3.0])
    const = np.array([5.0, 5.0, 5.0])
    assert spearman_rank_correlation(a, const) == 0.0


@pytest.mark.phase8
def test_spearman_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        spearman_rank_correlation(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


@pytest.mark.phase8
def test_compute_shap_importance_shape_and_sign() -> None:
    """SHAP importance over a fitted RandomForest is a non-negative
    per-feature vector. Feature 0 drives the label, so it should rank top."""
    rng = np.random.default_rng(0)
    n, p = 300, 4
    X = rng.standard_normal((n, p))
    y = (X[:, 0] + 0.2 * rng.standard_normal(n) > 0).astype(np.int64)
    model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=0, n_jobs=1)
    model.fit(X, y)
    importance = compute_shap_importance(model, X)
    assert importance.shape == (p,)
    assert np.all(importance >= 0.0)
    # The signal feature (index 0) must be the most important.
    assert int(np.argmax(importance)) == 0
