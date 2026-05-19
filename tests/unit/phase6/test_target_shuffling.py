"""Phase 6 — target shuffling leakage gate (Blueprint §8.3)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from afml.validation.target_shuffling import (
    DataLeakageError,
    target_shuffling_test,
)


def _signal_dataset(
    n: int = 400, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    return X, y, t0, t1


def _noise_dataset(
    n: int = 400, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """X has no relation to y."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = rng.integers(0, 2, size=n).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    return X, y, t0, t1


@pytest.mark.phase6
def test_target_shuffling_passes_real_signal() -> None:
    """Strong signal ⇒ real Brier ≪ shuffled Briers ⇒ small p-value ⇒ no leakage."""
    X, y, t0, t1 = _signal_dataset()
    result = target_shuffling_test(
        lambda: RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1),
        X,
        y,
        t0,
        t1,
        n_shuffles=10,
        n_splits=3,
        embargo_pct=0.02,
        random_state=0,
        raise_on_leakage=False,
    )
    assert result.brier_real < result.shuffled_mean - 0.05
    assert result.pvalue < 0.05


@pytest.mark.phase6
def test_target_shuffling_detects_no_signal_on_noise() -> None:
    """Pure noise ⇒ real Brier ≈ shuffled Brier ⇒ large p-value ⇒ leakage flag."""
    X, y, t0, t1 = _noise_dataset()
    result = target_shuffling_test(
        lambda: RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1),
        X,
        y,
        t0,
        t1,
        n_shuffles=10,
        n_splits=3,
        embargo_pct=0.02,
        random_state=0,
        raise_on_leakage=False,
    )
    # Real Brier should NOT be much better than shuffled mean.
    assert result.pvalue >= 0.05


@pytest.mark.phase6
def test_target_shuffling_raises_on_leakage_by_default() -> None:
    """On a noise dataset with raise_on_leakage=True (default), the test
    raises DataLeakageError."""
    X, y, t0, t1 = _noise_dataset()
    with pytest.raises(DataLeakageError):
        target_shuffling_test(
            lambda: RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1),
            X,
            y,
            t0,
            t1,
            n_shuffles=10,
            n_splits=3,
            embargo_pct=0.02,
            random_state=0,
            raise_on_leakage=True,
        )


@pytest.mark.phase6
def test_target_shuffling_reproducible_with_seed() -> None:
    X, y, t0, t1 = _signal_dataset()
    res1 = target_shuffling_test(
        lambda: RandomForestClassifier(n_estimators=15, random_state=0, n_jobs=1),
        X,
        y,
        t0,
        t1,
        n_shuffles=5,
        n_splits=3,
        random_state=42,
        raise_on_leakage=False,
    )
    res2 = target_shuffling_test(
        lambda: RandomForestClassifier(n_estimators=15, random_state=0, n_jobs=1),
        X,
        y,
        t0,
        t1,
        n_shuffles=5,
        n_splits=3,
        random_state=42,
        raise_on_leakage=False,
    )
    np.testing.assert_allclose(res1.brier_shuffled, res2.brier_shuffled)
    assert res1.pvalue == res2.pvalue


@pytest.mark.phase6
def test_target_shuffling_rejects_non_binary_y() -> None:
    X, _, t0, t1 = _signal_dataset()
    rng = np.random.default_rng(0)
    bad_y: np.ndarray = np.asarray(rng.integers(0, 3, size=X.shape[0]), dtype=np.int64)
    with pytest.raises(ValueError, match="binary"):
        target_shuffling_test(
            lambda: RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1),
            X,
            bad_y,
            t0,
            t1,
            n_shuffles=2,
            raise_on_leakage=False,
        )
