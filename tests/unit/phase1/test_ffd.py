"""Phase 1 — Fixed-Width Fractional Differencing (Blueprint §3.2)."""

from __future__ import annotations

import numpy as np
import pytest

from afml.data.ffd import ffd_apply, ffd_weights, find_optimal_d
from afml.data.stationarity import adf_pvalue


@pytest.mark.phase1
def test_weights_first_is_one() -> None:
    w = ffd_weights(0.5, tol=1e-5)
    assert w[0] == pytest.approx(1.0)


@pytest.mark.phase1
def test_weights_satisfy_recurrence() -> None:
    """``ω_k = -ω_{k-1} · (d - k + 1) / k`` exactly."""
    d = 0.4
    w = ffd_weights(d, tol=1e-6)
    for k in range(1, len(w)):
        expected = -w[k - 1] * (d - k + 1.0) / k
        assert w[k] == pytest.approx(expected, rel=1e-12)


@pytest.mark.phase1
def test_weights_terminate_below_tol() -> None:
    tol = 1e-4
    w = ffd_weights(0.5, tol=tol)
    # All retained weights have magnitude ≥ tol; the next weight would be < tol.
    assert np.all(np.abs(w) >= tol)
    next_w = -w[-1] * (0.5 - len(w) + 1.0) / len(w)
    assert abs(next_w) < tol


@pytest.mark.phase1
def test_weights_magnitudes_decay() -> None:
    """|ω_k| should decay (not strictly monotonic for all k due to sign-aware
    formula, but overall trend must decay since we terminate below tol)."""
    w = ffd_weights(0.5, tol=1e-6)
    # Tail decay: every weight beyond index 5 should be < |ω_0|.
    assert np.all(np.abs(w[5:]) < abs(w[0]))


@pytest.mark.phase1
def test_weights_rejects_invalid_d() -> None:
    with pytest.raises(ValueError, match=r"d must be in \(0, 1\]"):
        ffd_weights(0.0, tol=1e-5)
    with pytest.raises(ValueError, match=r"d must be in \(0, 1\]"):
        ffd_weights(1.5, tol=1e-5)


@pytest.mark.phase1
def test_weights_rejects_invalid_tol() -> None:
    with pytest.raises(ValueError, match="tol"):
        ffd_weights(0.5, tol=0.0)


@pytest.mark.phase1
def test_apply_drops_first_l_star_rows_no_nan() -> None:
    """AFML audit V2 — Constant History Length invariant.

    Output length must equal ``n - l*`` exactly, and ALL output values must be
    finite. There is no NaN warm-up: every emitted row uses exactly ``l*``
    input lags. We use d=0.8 here so the default-tolerance window stays well
    below the series length (l* ≈ 228 ≪ 3000).
    """
    rng = np.random.default_rng(123)
    series = np.cumsum(rng.standard_normal(3000))
    d = 0.8
    out = ffd_apply(series, d=d)
    w = ffd_weights(d)
    assert len(w) < len(series), "test setup requires l* < n"

    # Audit-mandated length identity.
    assert len(out) == len(series) - len(w)
    # No warm-up NaN — every point has a complete history.
    assert np.all(np.isfinite(out))


@pytest.mark.phase1
def test_apply_returns_empty_when_window_exceeds_input() -> None:
    """If ``n ≤ l*`` there are zero output points with a complete history."""
    rng = np.random.default_rng(0)
    short = np.cumsum(rng.standard_normal(50))  # well below any reasonable l*
    out = ffd_apply(short, d=0.4)  # d=0.4 gives l* > 1000
    assert out.shape == (0,)


@pytest.mark.phase1
def test_apply_d_equals_one_approximates_first_diff() -> None:
    """``d = 1`` gives weights ≈ (1, -1) — first difference.

    With the new constant-history convention, the output covers input indices
    ``[l*, n-1]``. For ``d=1`` and ``tol=1e-12`` the window is exactly 2, so
    output index 0 corresponds to ``series[2] - series[1]``, output index 1 to
    ``series[3] - series[2]``, etc.
    """
    series = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    out = ffd_apply(series, d=1.0, tol=1e-12)
    w = ffd_weights(1.0, tol=1e-12)
    n_w = len(w)
    # Output covers input indices [n_w, n-1] = [2, 4]
    diffs = np.diff(series)  # [2, 3, 4, 5]
    expected = diffs[n_w - 1 :]  # [3, 4, 5]
    assert out == pytest.approx(expected)


@pytest.mark.phase1
def test_apply_raises_on_2d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        ffd_apply(np.zeros((10, 2)), d=0.4)


@pytest.mark.phase1
def test_ffd_makes_random_walk_stationary() -> None:
    """Blueprint §3.3 DoD: ``adfuller(ffd_series)[1] < 0.05``."""
    rng = np.random.default_rng(2026)
    rw = np.cumsum(rng.standard_normal(3000))
    # The random walk itself is non-stationary:
    assert adf_pvalue(rw) > 0.05
    # FFD at d=0.5 should make it stationary.
    ffd = ffd_apply(rw, d=0.5)
    assert adf_pvalue(ffd) < 0.05


@pytest.mark.phase1
def test_find_optimal_d_returns_lowest_passing_grid_point() -> None:
    rng = np.random.default_rng(7)
    rw = np.cumsum(rng.standard_normal(3000))
    result = find_optimal_d(rw, d_grid=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    # Some d in the grid must pass.
    assert result.d_optimal is not None
    assert result.adf_pvalue_at_optimum is not None
    assert result.adf_pvalue_at_optimum < 0.05
    # All smaller grid points than the chosen one must NOT pass.
    smaller = [d for d in result.sweep if d < result.d_optimal]
    for d in smaller:
        assert not (result.sweep[d] < 0.05)


@pytest.mark.phase1
def test_find_optimal_d_audit_log_complete() -> None:
    """``FFDResult.sweep`` records every grid point tried (PRD §4 anti-lazy)."""
    rng = np.random.default_rng(99)
    rw = np.cumsum(rng.standard_normal(2000))
    grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    result = find_optimal_d(rw, d_grid=grid)
    assert set(result.sweep.keys()) == set(grid.tolist())


@pytest.mark.phase1
def test_find_optimal_d_returns_window_length() -> None:
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.standard_normal(3000))
    result = find_optimal_d(rw, d_grid=np.array([0.5]))
    if result.d_optimal is not None:
        assert result.window_length is not None
        # Window equals weight count.
        assert result.window_length == len(ffd_weights(result.d_optimal))
