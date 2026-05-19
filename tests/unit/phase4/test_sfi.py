"""Phase 4 — Single Feature Importance fallback."""

from __future__ import annotations

import numpy as np
import pytest

from afml.selection.sfi import single_feature_importance


@pytest.mark.phase4
def test_sfi_promotes_signal_feature() -> None:
    """A pure-signal feature must score above pure-noise features."""
    rng = np.random.default_rng(0)
    n = 800
    latent = rng.standard_normal(n)
    y = (latent > 0.0).astype(np.int64)
    cols = [
        latent + rng.standard_normal(n) * 0.2,  # signal
        rng.standard_normal(n),  # noise
        rng.standard_normal(n),  # noise
        rng.standard_normal(n),  # noise
    ]
    X = np.column_stack(cols)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    result = single_feature_importance(X, y, t0, t1, n_splits=4, random_state=0)
    # Signal column (index 0) must outrank the others.
    sorted_features = sorted(result.feature_importance.items(), key=lambda kv: kv[1], reverse=True)
    assert sorted_features[0][0] == 0
    # And it must clear the survival threshold.
    assert 0 in result.surviving_columns


@pytest.mark.phase4
def test_sfi_drops_pure_noise_features() -> None:
    """Every feature uncorrelated with ``y`` should fail the survival
    threshold (or at worst be only marginally above zero)."""
    rng = np.random.default_rng(0)
    n = 600
    X = rng.standard_normal((n, 5))
    y = rng.integers(0, 2, size=n)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    result = single_feature_importance(X, y, t0, t1, n_splits=4, random_state=0)
    # No feature should score appreciably above zero.
    for j, imp in result.feature_importance.items():
        assert imp < 0.05, f"noise feature {j} has unexpectedly high SFI score: {imp:.4f}"


@pytest.mark.phase4
def test_sfi_handles_degenerate_fold_gracefully() -> None:
    """A tiny test fold with one class must not crash the pipeline."""
    rng = np.random.default_rng(0)
    n = 40
    X = rng.standard_normal((n, 3))
    y = np.concatenate([np.zeros(20, dtype=np.int64), np.ones(20, dtype=np.int64)])
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0
    # n_splits = 5 ⇒ each fold has 8 rows; with the y split above, some folds
    # may be class-degenerate. The SFI must skip those gracefully.
    result = single_feature_importance(X, y, t0, t1, n_splits=5, random_state=0)
    assert len(result.feature_importance) == 3
    assert np.all(np.isfinite(list(result.feature_importance.values())))


@pytest.mark.phase4
def test_sfi_reproducible_with_same_seed() -> None:
    rng = np.random.default_rng(0)
    n = 400
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 1
    r1 = single_feature_importance(X, y, t0, t1, random_state=5)
    r2 = single_feature_importance(X, y, t0, t1, random_state=5)
    assert r1.feature_importance == r2.feature_importance
    assert r1.surviving_columns == r2.surviving_columns
