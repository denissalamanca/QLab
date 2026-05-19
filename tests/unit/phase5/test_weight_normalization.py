"""Phase 5 pre-Phase-6 patch — uniqueness-weight normalisation.

Raw average-uniqueness weights ``ū_i`` are fractions in ``(0, 1]``. Passed
directly to XGBoost, they can trigger ``min_child_weight``-driven vanishing
gradients (a leaf needs ``Σ ū ≥ min_child_weight`` to be considered, so
fractional weights demand more samples per leaf than necessary).

The patch (``afml.modeling.calibration._normalize_sample_weights``) scales
the input vector so the aggregate weight equals the sample count ``N``,
preserving the per-sample weight ratios. These tests lock in:

1. The post-normalisation sum equals ``N`` exactly (no float drift).
2. Per-sample ratios are preserved.
3. Degenerate input (all zeros) returns uniform weights.
4. The normalisation is invoked by both calibration entry points.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from afml.modeling import calibration as calibration_module
from afml.modeling.calibration import _normalize_sample_weights


@pytest.mark.phase5
def test_normalization_sums_to_n() -> None:
    """The post-normalisation weight vector sums to exactly N."""
    rng = np.random.default_rng(0)
    n = 250
    w = rng.uniform(0.05, 1.0, size=n)
    normalized = _normalize_sample_weights(w)
    assert np.isclose(normalized.sum(), float(n))
    assert normalized.shape == w.shape
    assert normalized.dtype == np.float64


@pytest.mark.phase5
def test_normalization_preserves_per_sample_ratios() -> None:
    """``normalized[i] / normalized[j] == w[i] / w[j]`` for all i, j."""
    rng = np.random.default_rng(0)
    w = rng.uniform(0.1, 1.0, size=100)
    normalized = _normalize_sample_weights(w)
    # Compare a representative subset of pairwise ratios.
    pairs = [(0, 50), (1, 99), (25, 75), (10, 90)]
    for i, j in pairs:
        np.testing.assert_allclose(normalized[i] / normalized[j], w[i] / w[j])


@pytest.mark.phase5
def test_normalization_handles_zero_input() -> None:
    """All-zero weights should yield uniform ones rather than NaN."""
    n = 30
    w = np.zeros(n, dtype=np.float64)
    normalized = _normalize_sample_weights(w)
    np.testing.assert_array_equal(normalized, np.ones(n, dtype=np.float64))
    assert normalized.sum() == float(n)


@pytest.mark.phase5
def test_normalization_idempotent_on_already_normalised_input() -> None:
    """A vector that already sums to N is unchanged."""
    n = 50
    w = np.ones(n, dtype=np.float64)
    normalized = _normalize_sample_weights(w)
    np.testing.assert_allclose(normalized, w)


@pytest.mark.phase5
def test_normalization_handles_extreme_skew() -> None:
    """A weight vector dominated by one large value still normalises cleanly."""
    n = 100
    w = np.full(n, 1e-6)
    w[0] = 10.0
    normalized = _normalize_sample_weights(w)
    assert np.isclose(normalized.sum(), float(n))
    # Largest entry stays largest after scaling.
    assert int(np.argmax(normalized)) == 0


@pytest.mark.phase5
def test_calibration_modules_invoke_normalization() -> None:
    """AST-level audit — both purged-CV calibration entry points must call
    ``_normalize_sample_weights`` on the input weights before passing them
    down to the underlying ``fit``."""
    src = inspect.getsource(calibration_module)
    tree = ast.parse(src)
    target_functions = {
        "fit_calibrated_sbrf_with_purged_cv",
        "fit_calibrated_classifier_with_purged_cv",
    }
    invocations_by_func: dict[str, int] = dict.fromkeys(target_functions, 0)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in target_functions:
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "_normalize_sample_weights"
                ):
                    invocations_by_func[node.name] += 1
    for func_name, count in invocations_by_func.items():
        assert count >= 1, (
            f"{func_name} does not invoke _normalize_sample_weights — "
            f"XGBoost min_child_weight squash is unguarded"
        )
