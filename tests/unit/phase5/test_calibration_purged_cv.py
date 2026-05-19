"""Phase 5 audit V1 — calibration via PurgedKFold cross-fitting.

The Lead Quant flagged that ``sklearn.calibration.CalibratedClassifierCV``
silently uses stratified KFold when ``cv`` defaults are accepted. That randomly
shuffles non-IID financial labels and leaks future information into the
calibrator. The remediation: every calibration path must explicitly invoke
``PurgedKFold.split``.

The tests below lock that contract in:

1. The SBRF path's ``fit_calibrated_sbrf_with_purged_cv`` runs a manual
   cross-fitting loop and we verify ``PurgedKFold.split`` is the entry
   point — by patching the class method with a spy and asserting it was
   called.
2. The generic-classifier path's ``fit_calibrated_classifier_with_purged_cv``
   funnels through ``CalibratedClassifierCV(cv=PurgedKFoldSklearn(...))``;
   we verify the wrapper's ``.split`` is invoked during the calibrator's
   ``.fit()``.
"""

from __future__ import annotations

import ast
import inspect
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from afml.modeling import (
    SequentiallyBootstrappedRandomForest,
    fit_calibrated_classifier_with_purged_cv,
    fit_calibrated_sbrf_with_purged_cv,
    indicator_matrix,
)
from afml.modeling import calibration as calibration_module
from afml.modeling.calibration import PurgedCVCalibratedSBRF
from afml.selection.purged_kfold import PurgedKFold, PurgedKFoldSklearn


def _synthetic_fit_inputs(
    *, n: int = 800, n_features: int = 4, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = (X[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 3
    ind = indicator_matrix(t0, t1)
    sw = rng.uniform(0.5, 1.0, size=n)
    return X, y, t0, t1, ind, sw


@pytest.mark.phase5
def test_sbrf_calibration_invokes_purged_kfold_split() -> None:
    """AFML 0-5 audit V1 — ``PurgedKFold.split`` must be called during the
    SBRF cross-fitting calibration. The test patches the class method with a
    spy and asserts it was hit at least once."""
    X, y, t0, t1, ind, sw = _synthetic_fit_inputs(n=400)
    n_train = 300
    sbrf = SequentiallyBootstrappedRandomForest(n_estimators=15, random_state=0)

    real_split = PurgedKFold.split
    call_log: list[tuple[int, int]] = []

    def spy_split(self: PurgedKFold, t0_arr: np.ndarray, t1_arr: np.ndarray):  # type: ignore[no-untyped-def]
        call_log.append((t0_arr.size, t1_arr.size))
        return real_split(self, t0_arr, t1_arr)

    with patch.object(PurgedKFold, "split", new=spy_split):
        result = fit_calibrated_sbrf_with_purged_cv(
            sbrf,
            X[:n_train],
            y[:n_train],
            ind[:, :n_train],
            t0[:n_train],
            t1[:n_train],
            sw[:n_train],
            X[n_train:],
            y[n_train:],
            n_splits=3,
            embargo_pct=0.02,
        )
    assert call_log, "PurgedKFold.split was never invoked during SBRF calibration"
    # Each call should have received the n_train-length horizon arrays.
    assert all(n_t0 == n_train and n_t1 == n_train for n_t0, n_t1 in call_log)
    assert isinstance(result.calibrated, PurgedCVCalibratedSBRF)
    assert result.method in ("isotonic", "sigmoid")


@pytest.mark.phase5
def test_generic_classifier_calibration_uses_purged_kfold_sklearn() -> None:
    """AFML 0-5 audit V1 — the XGBoost / vanilla-RF path threads
    ``PurgedKFoldSklearn`` into ``CalibratedClassifierCV(cv=...)``. We spy
    on the wrapper's ``split`` method and confirm it was invoked."""
    X, y, t0, t1, _ind, sw = _synthetic_fit_inputs(n=400)
    n_train = 300
    rf = RandomForestClassifier(n_estimators=10, random_state=0)

    real_split = PurgedKFoldSklearn.split
    invocations: list[bool] = []

    def spy_split(self: PurgedKFoldSklearn, *args, **kwargs):  # type: ignore[no-untyped-def]
        invocations.append(True)
        return real_split(self, *args, **kwargs)

    with patch.object(PurgedKFoldSklearn, "split", new=spy_split):
        result = fit_calibrated_classifier_with_purged_cv(
            rf,
            X[:n_train],
            y[:n_train],
            t0[:n_train],
            t1[:n_train],
            sw[:n_train],
            X[n_train:],
            y[n_train:],
            n_splits=3,
            embargo_pct=0.02,
        )
    assert invocations, "PurgedKFoldSklearn.split was never invoked"
    assert result.method in ("isotonic", "sigmoid")


@pytest.mark.phase5
def test_purged_kfold_sklearn_satisfies_sklearn_cv_interface() -> None:
    """The adapter must expose ``split(X, y, groups)`` and ``get_n_splits``
    in the shape sklearn expects from a CV object."""
    t0 = np.arange(100, dtype=np.int64)
    t1 = t0 + 2
    cv = PurgedKFoldSklearn(t0, t1, n_splits=4, embargo_pct=0.01)

    assert cv.get_n_splits() == 4
    # Calling split with X / y / groups must not raise.
    folds = list(cv.split(X=np.zeros((100, 3)), y=np.zeros(100), groups=None))
    assert len(folds) == 4
    # Each yielded pair is (train_idx, test_idx) disjoint int arrays.
    for tr, te in folds:
        assert tr.dtype == np.int64
        assert te.dtype == np.int64
        assert np.intersect1d(tr, te).size == 0


@pytest.mark.phase5
def test_purged_kfold_sklearn_rejects_mismatched_t0_t1() -> None:
    with pytest.raises(ValueError, match="same shape"):
        PurgedKFoldSklearn(np.arange(10), np.arange(5))


@pytest.mark.phase5
def test_calibration_module_banned_sklearn_kfold_paths() -> None:
    """Static AST guard — the calibration module must not import the
    banned ``KFold`` / ``TimeSeriesSplit`` from sklearn.model_selection."""
    src = inspect.getsource(calibration_module)
    tree = ast.parse(src)
    banned = {
        ("sklearn.model_selection", "KFold"),
        ("sklearn.model_selection", "TimeSeriesSplit"),
        ("sklearn.model_selection", "StratifiedKFold"),
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                assert (module, alias.name) not in banned, (
                    f"banned import {module}.{alias.name} in calibration.py"
                )
