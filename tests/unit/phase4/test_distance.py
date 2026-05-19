"""Phase 4 — AFML correlation-distance metric."""

from __future__ import annotations

import numpy as np
import pytest

from afml.selection.distance import afml_distance_matrix


@pytest.mark.phase4
def test_distance_is_symmetric() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6))
    D = afml_distance_matrix(X)
    np.testing.assert_array_equal(D, D.T)


@pytest.mark.phase4
def test_distance_has_zero_diagonal() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6))
    D = afml_distance_matrix(X)
    np.testing.assert_array_equal(np.diag(D), np.zeros(6))


@pytest.mark.phase4
def test_distance_bounded_zero_one() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 8))
    D = afml_distance_matrix(X)
    assert D.min() >= 0.0
    assert D.max() <= 1.0 + 1e-12


@pytest.mark.phase4
def test_distance_identical_columns_is_zero() -> None:
    """Two identical columns ⇒ ρ = 1 ⇒ d = 0.

    Tolerance is ``1e-6`` rather than machine epsilon because the path is
    ``corrcoef → 1.0 (up to ULP) → sqrt(0.5 * 1e-16) ≈ 1e-8``. The exponent
    cascade through the square root rules out tighter tolerances on float64.
    """
    rng = np.random.default_rng(0)
    col = rng.standard_normal(300)
    X = np.column_stack([col, col, rng.standard_normal(300)])
    D = afml_distance_matrix(X)
    assert D[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert D[1, 0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.phase4
def test_distance_anticorrelated_columns_is_one() -> None:
    """Perfectly anti-correlated columns ⇒ ρ = −1 ⇒ d = 1."""
    rng = np.random.default_rng(0)
    col = rng.standard_normal(300)
    X = np.column_stack([col, -col, rng.standard_normal(300)])
    D = afml_distance_matrix(X)
    # Same ULP cascade as the identical-columns case (1e-6 is plenty).
    assert D[0, 1] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.phase4
def test_distance_independent_columns_close_to_sqrt_half() -> None:
    """Independent ⇒ ρ ≈ 0 ⇒ d ≈ √0.5 ≈ 0.7071."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10000, 4))
    D = afml_distance_matrix(X)
    off_diag = D[~np.eye(4, dtype=bool)]
    expected = np.sqrt(0.5)
    np.testing.assert_allclose(off_diag, expected, atol=0.05)


@pytest.mark.phase4
def test_distance_rejects_nan() -> None:
    X = np.ones((10, 3))
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"NaN|Inf"):
        afml_distance_matrix(X)


@pytest.mark.phase4
def test_distance_rejects_too_small_input() -> None:
    with pytest.raises(ValueError, match=r"≥ 2"):
        afml_distance_matrix(np.ones((1, 3)))
    with pytest.raises(ValueError, match=r"≥ 2"):
        afml_distance_matrix(np.ones((10, 1)))


@pytest.mark.phase4
def test_distance_handles_constant_column_without_nan() -> None:
    """A constant column has undefined correlation. The implementation must
    NOT propagate NaN — it should treat constants as maximally-uncertain
    (d = √0.5) rather than blow up Ward clustering downstream."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4))
    X[:, 2] = 5.0  # constant
    D = afml_distance_matrix(X)
    assert np.all(np.isfinite(D))
    expected = np.sqrt(0.5)
    np.testing.assert_allclose(D[2, :2], expected, atol=1e-9)
    np.testing.assert_allclose(D[2, 3], expected, atol=1e-9)
    assert D[2, 2] == 0.0


@pytest.mark.phase4
def test_distance_triangle_inequality_holds() -> None:
    """``d`` is a proper metric ⇒ d(a, c) ≤ d(a, b) + d(b, c)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 6))
    D = afml_distance_matrix(X)
    n = D.shape[0]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Allow a tiny tolerance for floating drift.
                assert D[i, k] <= D[i, j] + D[j, k] + 1e-9, (
                    f"triangle inequality violated for ({i}, {j}, {k})"
                )
