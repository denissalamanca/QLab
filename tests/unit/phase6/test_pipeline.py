"""Phase 6 — orchestrator + Blueprint §8.3 DoD.

Blueprint §8.3 DoD (verbatim):

* **PBO Constraint:** ``assert PBO_score < 0.05``.
* **Target Shuffling Leakage Test:** Shuffle ``y`` randomly, retrain. If the
  model retains predictive power on the shuffled data, raise
  ``DataLeakageError``.

Both gates are validated end-to-end on a controlled synthetic dataset
where one strategy carries real signal and a population of weak / noise
strategies provide the PBO contrast.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from afml.validation import DataLeakageError, validate_strategy


class _FeatureSubsetClassifier:
    """RF + a per-strategy feature index slice.

    Used to manufacture a candidate set with **deterministic IS/OOS rank
    ordering** — strategies that see the signal column outrank ones that
    see only noise, both IS and OOS, producing a low PBO.
    """

    def __init__(self, indices: list[int]) -> None:
        self.indices = list(indices)
        self._rf = RandomForestClassifier(n_estimators=25, max_depth=3, random_state=0, n_jobs=1)
        self.classes_: np.ndarray = np.asarray([0, 1], dtype=np.int64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _FeatureSubsetClassifier:
        self._rf.fit(X[:, self.indices], y)
        self.classes_ = self._rf.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self._rf.predict_proba(X[:, self.indices])
        return np.asarray(proba, dtype=np.float64)


def _strong_signal_setup() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Callable[[], Any]]
]:
    """Synthetic-signal dataset + a candidate set with **clear rank ordering**.

    The label depends entirely on ``X[:, 0]``. Strategies that include column
    0 see the signal; strategies that don't see only noise. This creates a
    bimodal OOS-Brier distribution where IS-best and OOS-best agree, so
    PBO ≪ 0.05.
    """
    rng = np.random.default_rng(20260605)
    n = 800
    X = rng.standard_normal((n, 6))
    latent = X[:, 0] + 0.3 * rng.standard_normal(n)
    y = (latent > 0.0).astype(np.int64)
    realized = np.where(y == 1, 0.01, -0.005) + 0.002 * rng.standard_normal(n)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 3
    candidates: list[Callable[[], Any]] = [
        lambda: _FeatureSubsetClassifier([0]),  # signal only — best
        lambda: _FeatureSubsetClassifier([0, 1]),  # signal + 1 noise — good
        lambda: _FeatureSubsetClassifier([0, 2]),  # signal + 1 noise — good
        lambda: _FeatureSubsetClassifier([1, 2, 3]),  # noise only — bad
        lambda: _FeatureSubsetClassifier([3, 4, 5]),  # noise only — bad
    ]
    return X, y, t0, t1, realized, candidates


@pytest.mark.phase6
def test_pipeline_pbo_below_threshold_on_signal_data() -> None:
    """Blueprint §8.3 — PBO < 0.05 on a dataset with real signal and a
    candidate set that includes the signal-bearing strategy."""
    X, y, t0, t1, realized, candidates = _strong_signal_setup()
    result = validate_strategy(
        candidates,
        X,
        y,
        t0,
        t1,
        realized,
        n_trials=20,
        n_groups=6,
        n_test_groups=2,
        embargo_pct=0.01,
        n_shuffles=5,
        random_state=0,
        raise_on_leakage=False,
    )
    assert result.pbo.pbo < 0.05, f"PBO {result.pbo.pbo:.3f} ≥ 0.05 on signal data"


@pytest.mark.phase6
def test_pipeline_target_shuffling_passes_on_real_signal() -> None:
    """The orchestrator's embedded shuffling step must NOT flag leakage on a
    genuine-signal dataset."""
    X, y, t0, t1, realized, candidates = _strong_signal_setup()
    result = validate_strategy(
        candidates,
        X,
        y,
        t0,
        t1,
        realized,
        n_trials=20,
        n_shuffles=10,
        n_groups=6,
        n_test_groups=2,
        random_state=0,
        raise_on_leakage=False,
    )
    assert result.target_shuffling.pvalue < 0.05


@pytest.mark.phase6
def test_pipeline_raises_on_pure_noise() -> None:
    """When labels are pure noise, the shuffling-test gate trips. Defaults
    have raise_on_leakage=True, so the orchestrator surfaces the error.

    We use a large shuffle count (30) so the empirical p-value is stable
    around 0.5 on noise — with n_shuffles=5 the p-value can land below
    0.05 by chance even on truly random data.
    """
    rng = np.random.default_rng(0)
    n = 600
    X = rng.standard_normal((n, 5))
    y = rng.integers(0, 2, size=n).astype(np.int64)
    realized = rng.standard_normal(n) * 0.01
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    candidates: list[Callable[[], Any]] = [
        lambda: RandomForestClassifier(n_estimators=15, max_depth=3, random_state=0, n_jobs=1),
        lambda: DecisionTreeClassifier(max_depth=3, random_state=1),
    ]
    with pytest.raises(DataLeakageError):
        validate_strategy(
            candidates,
            X,
            y,
            t0,
            t1,
            realized,
            n_trials=5,
            n_groups=4,
            n_test_groups=2,
            n_shuffles=30,
            random_state=0,
            raise_on_leakage=True,
        )


@pytest.mark.phase6
def test_pipeline_passes_full_phase6_dod() -> None:
    """All three gates together — PBO + DSR + target shuffling.

    Uses a feature-subset candidate set with deterministic rank ordering
    (signal-bearing strategies dominate noise-only ones) so PBO is near
    zero. A realistic ``sharpe_std_of_trials=0.3`` prior (instead of the
    conservative 1.0 default) makes the DSR deflation tractable on this
    synthetic signal.
    """
    X, y, t0, t1, realized, candidates = _strong_signal_setup()
    result = validate_strategy(
        candidates,
        X,
        y,
        t0,
        t1,
        realized,
        # n_trials=50 ≥ DSR cold-start threshold (30), so the DSR is computed
        # rather than quarantined (AFML Phase 0-6 audit V2).
        n_trials=50,
        n_groups=6,
        n_test_groups=2,
        embargo_pct=0.01,
        n_shuffles=20,
        sharpe_std_of_trials=0.3,
        random_state=0,
        raise_on_leakage=False,
        dsr_threshold=0.5,
    )
    assert not result.dsr.quarantined
    assert result.passes_phase6_dod, (
        f"PBO={result.pbo.pbo:.3f}, DSR={result.dsr.dsr:.3f}, "
        f"shuffle_p={result.target_shuffling.pvalue:.3f}"
    )


@pytest.mark.phase6
def test_pipeline_quarantines_on_cold_start() -> None:
    """AFML Phase 0-6 audit V2 — with fewer than 30 trials the DSR is
    quarantined and the strategy fails the Phase 6 gate regardless of PBO /
    leakage outcomes."""
    X, y, t0, t1, realized, candidates = _strong_signal_setup()
    result = validate_strategy(
        candidates,
        X,
        y,
        t0,
        t1,
        realized,
        n_trials=5,  # ≪ 30 → cold-start quarantine
        n_groups=6,
        n_test_groups=2,
        n_shuffles=10,
        sharpe_std_of_trials=0.3,
        random_state=0,
        raise_on_leakage=False,
    )
    assert result.dsr.quarantined is True
    assert result.dsr.dsr == 0.0
    assert result.passes_phase6_dod is False


@pytest.mark.phase6
def test_pipeline_rejects_too_few_candidates() -> None:
    X, y, t0, t1, realized, _ = _strong_signal_setup()
    with pytest.raises(ValueError, match="≥ 2"):
        validate_strategy(
            [lambda: RandomForestClassifier(n_estimators=5, random_state=0)],
            X,
            y,
            t0,
            t1,
            realized,
            n_trials=5,
            raise_on_leakage=False,
        )
