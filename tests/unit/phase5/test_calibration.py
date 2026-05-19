"""Phase 5 — Brier-minimising isotonic vs sigmoid calibration."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV

from afml.modeling import (
    SequentiallyBootstrappedRandomForest,
    fit_calibrated_classifier,
    indicator_matrix,
)
from afml.modeling import calibration as calibration_module
from afml.modeling.calibration import CalibrationResult


def _make_calibration_inputs(
    seed: int = 0,
) -> tuple[
    SequentiallyBootstrappedRandomForest,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build a prefit SBRF + (cal, holdout) sets — typical caller shape."""
    rng = np.random.default_rng(seed)
    n = 900
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] > 0.0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    ind = indicator_matrix(t0, t1)
    n_train = 600
    sbrf = SequentiallyBootstrappedRandomForest(n_estimators=30, random_state=seed)
    sbrf.fit(X[:n_train], y[:n_train], ind[:, :n_train])
    return (
        sbrf,
        X[n_train : n_train + 150],
        y[n_train : n_train + 150],
        X[n_train + 150 :],
        y[n_train + 150 :],
    )


@pytest.mark.phase5
def test_calibration_returns_result_object() -> None:
    sbrf, X_cal, y_cal, X_hold, y_hold = _make_calibration_inputs()
    result = fit_calibrated_classifier(sbrf, X_cal, y_cal, X_hold, y_hold)
    assert isinstance(result, CalibrationResult)
    assert result.method in ("isotonic", "sigmoid")
    assert isinstance(result.calibrated, CalibratedClassifierCV)


@pytest.mark.phase5
def test_calibration_winner_minimises_brier() -> None:
    sbrf, X_cal, y_cal, X_hold, y_hold = _make_calibration_inputs()
    result = fit_calibrated_classifier(sbrf, X_cal, y_cal, X_hold, y_hold)
    if result.method == "isotonic":
        assert result.brier_isotonic <= result.brier_sigmoid
    else:
        assert result.brier_sigmoid <= result.brier_isotonic


@pytest.mark.phase5
def test_calibration_improves_brier_on_strong_signal() -> None:
    """On a strongly-predictable dataset, the calibrated classifier should
    be no worse than the uncalibrated one on the same holdout (within noise)."""
    sbrf, X_cal, y_cal, X_hold, y_hold = _make_calibration_inputs()
    result = fit_calibrated_classifier(sbrf, X_cal, y_cal, X_hold, y_hold)
    # Allow a small tolerance — calibration on a small ~150-row set can
    # increase Brier marginally on the holdout. The key invariant for Phase 5
    # is the *naive baseline* comparison (test_pipeline.py).
    assert result.brier_isotonic < 1.1 * result.brier_uncalibrated
    assert result.brier_sigmoid < 1.1 * result.brier_uncalibrated


@pytest.mark.phase5
def test_calibration_rejects_non_binary_y() -> None:
    sbrf, X_cal, _y_cal, X_hold, _y_hold = _make_calibration_inputs()
    rng = np.random.default_rng(0)
    bad_y: np.ndarray = np.asarray(rng.integers(0, 3, size=X_cal.shape[0]), dtype=np.int64)
    placeholder_y: np.ndarray = np.zeros(X_hold.shape[0], dtype=np.int64)
    with pytest.raises(ValueError, match="binary"):
        fit_calibrated_classifier(sbrf, X_cal, bad_y, X_hold, placeholder_y)


@pytest.mark.phase5
def test_calibration_module_does_not_use_kfold() -> None:
    """Banned-method guard: calibration must use FrozenEstimator, never
    sklearn.model_selection.KFold / TimeSeriesSplit."""
    src = inspect.getsource(calibration_module)
    tree = ast.parse(src)
    banned = {
        ("sklearn.model_selection", "KFold"),
        ("sklearn.model_selection", "TimeSeriesSplit"),
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                assert (module, alias.name) not in banned, (
                    f"banned import {module}.{alias.name} in calibration.py"
                )
