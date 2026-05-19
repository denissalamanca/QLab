"""Phase 5 audit V2.1 — empty-MDA pass-through.

If Phase 4's Clustered MDA returns zero surviving features (circuit breaker
fired), Phase 5 must NOT crash. It returns a sentinel ``BrainTwoResult`` with
``halted_at_mda_upstream=True``, no fold diagnostics, and a ``None``
calibration artefact. The Phase 4 → Phase 5 hand-off then survives the
empty-matrix case cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from afml.modeling import train_brain_two


@pytest.mark.phase5
def test_empty_feature_matrix_returns_sentinel_result() -> None:
    """``X.shape[1] == 0`` ⇒ halted_at_mda_upstream, no fits attempted."""
    n = 500
    X_empty = np.zeros((n, 0), dtype=np.float64)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2

    result = train_brain_two(X_empty, y, t0, t1, n_splits=3, random_state=0)
    assert result.halted_at_mda_upstream is True
    assert result.fold_diagnostics == []
    assert result.last_calibration is None
    # DoD must report False when halted upstream — the caller cannot deploy.
    assert result.passes_phase5_dod is False


@pytest.mark.phase5
def test_zero_rows_returns_sentinel_result() -> None:
    """Defensive: ``X.shape[0] == 0`` also short-circuits cleanly."""
    X = np.zeros((0, 5), dtype=np.float64)
    y = np.zeros(0, dtype=np.int64)
    t0 = np.zeros(0, dtype=np.int64)
    t1 = np.zeros(0, dtype=np.int64)
    result = train_brain_two(X, y, t0, t1, n_splits=3, random_state=0)
    assert result.halted_at_mda_upstream is True
    assert result.last_calibration is None


@pytest.mark.phase5
def test_non_empty_input_still_runs_normally() -> None:
    """Sanity: with proper inputs, train_brain_two does NOT short-circuit."""
    rng = np.random.default_rng(0)
    n = 600
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    result = train_brain_two(
        X, y, t0, t1, n_splits=3, n_estimators=20, compare_with_xgboost=False, random_state=0
    )
    assert result.halted_at_mda_upstream is False
    assert len(result.fold_diagnostics) > 0
    assert result.last_calibration is not None


@pytest.mark.phase5
def test_outer_embargo_index_intersection_enforced_strictly() -> None:
    """AFML 0-5 audit V2 — every outer fold satisfies
    ``max(train_t1) + embargo < min(holdout_t0)``. With the realized
    ``t1`` arrays passed to ``train_brain_two``, this is the literal
    Blueprint §7.3 assertion."""
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 5  # heavy overlap so the embargo matters
    result = train_brain_two(
        X,
        y,
        t0,
        t1,
        n_splits=4,
        embargo_pct=0.02,
        n_estimators=20,
        compare_with_xgboost=False,
        random_state=0,
    )
    assert not result.halted_at_mda_upstream
    for fd in result.fold_diagnostics:
        assert fd.passes_index_intersection, (
            f"fold {fd.fold_index}: train_t1_max={fd.train_t1_max}, "
            f"embargo={fd.embargo_size}, holdout_t0_min={fd.holdout_t0_min}"
        )
