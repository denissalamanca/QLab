"""Phase 5 audit V3 — sample-weight propagation through SBRF + XGBoost.

The audit requires that the average-uniqueness vector ``ū_i`` actually
influences the trained model — i.e. that ``sample_weight`` is not being
silently dropped by any layer of the calibration stack.

Verification strategy: train two otherwise-identical estimators, one with
``sample_weight`` set to a deliberately non-uniform vector that zeroes out
half the rows, the other with uniform weights. If sample_weight propagates,
their predictions must differ.
"""

from __future__ import annotations

import numpy as np
import pytest
from xgboost import XGBClassifier

from afml.modeling import (
    SequentiallyBootstrappedRandomForest,
    fit_calibrated_classifier_with_purged_cv,
    fit_calibrated_sbrf_with_purged_cv,
    indicator_matrix,
)


@pytest.mark.phase5
def test_sbrf_sample_weight_influences_predictions() -> None:
    """An SBRF fit with non-uniform sample_weight must produce different
    predictions than one with uniform weights — confirming the weight
    actually reaches the underlying tree splits."""
    rng = np.random.default_rng(0)
    n = 600
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    ind = indicator_matrix(t0, t1)

    # Heavily skewed weight — first half gets weight 1, second half gets
    # weight 100, so the trees should fit the second half far more tightly.
    skewed = np.concatenate([np.ones(n // 2), np.full(n // 2, 100.0)])
    uniform = np.ones(n)

    sbrf_a = SequentiallyBootstrappedRandomForest(n_estimators=30, random_state=0)
    sbrf_a.fit(X, y, ind, sample_weight=skewed)
    sbrf_b = SequentiallyBootstrappedRandomForest(n_estimators=30, random_state=0)
    sbrf_b.fit(X, y, ind, sample_weight=uniform)

    proba_a = sbrf_a.predict_proba(X)
    proba_b = sbrf_b.predict_proba(X)
    # Predictions must differ by more than a numerical tolerance somewhere.
    assert np.max(np.abs(proba_a - proba_b)) > 1e-3, (
        "sample_weight had no effect on SBRF predictions — weight propagation broken"
    )


@pytest.mark.phase5
def test_sbrf_purged_calibration_sample_weight_propagates() -> None:
    """The purged-CV calibration pipeline must pass sample_weight all the
    way through to the underlying SBRF fits on every fold.

    Note: post-calibration probabilities saturate near 0 / 1 on confident
    predictions, which can mask weight effects. We test the *raw* SBRF
    inside the calibration wrapper to see the propagation directly.
    """
    rng = np.random.default_rng(0)
    n = 600
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    ind = indicator_matrix(t0, t1)

    n_tr = 450
    skewed = np.concatenate([np.ones(225), np.full(225, 100.0)])
    uniform = np.ones(n_tr)

    sbrf_template = SequentiallyBootstrappedRandomForest(n_estimators=20, random_state=0)
    res_a = fit_calibrated_sbrf_with_purged_cv(
        sbrf_template,
        X[:n_tr],
        y[:n_tr],
        ind[:, :n_tr],
        t0[:n_tr],
        t1[:n_tr],
        skewed,
        X[n_tr:],
        y[n_tr:],
        n_splits=3,
        embargo_pct=0.02,
    )
    res_b = fit_calibrated_sbrf_with_purged_cv(
        sbrf_template,
        X[:n_tr],
        y[:n_tr],
        ind[:, :n_tr],
        t0[:n_tr],
        t1[:n_tr],
        uniform,
        X[n_tr:],
        y[n_tr:],
        n_splits=3,
        embargo_pct=0.02,
    )

    # Drill into the wrapped SBRF — its raw predictions reflect sample_weight
    # most directly (the calibrator saturates near 0/1 on confident inputs).
    raw_a = res_a.calibrated.base_estimator.predict_proba(X[n_tr:])
    raw_b = res_b.calibrated.base_estimator.predict_proba(X[n_tr:])
    assert np.max(np.abs(raw_a - raw_b)) > 1e-3, (
        "sample_weight did not influence the calibrated SBRF's base estimator — propagation broken"
    )


@pytest.mark.phase5
def test_xgboost_purged_calibration_sample_weight_propagates() -> None:
    """Same check for the XGBoost calibration path — sample_weight must
    influence the holdout predictions."""
    rng = np.random.default_rng(0)
    n = 600
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2

    n_tr = 450
    skewed = np.concatenate([np.ones(225), np.full(225, 100.0)])
    uniform = np.ones(n_tr)

    xgb_template = XGBClassifier(n_estimators=30, max_depth=3, random_state=0, verbosity=0)
    res_a = fit_calibrated_classifier_with_purged_cv(
        xgb_template,
        X[:n_tr],
        y[:n_tr],
        t0[:n_tr],
        t1[:n_tr],
        skewed,
        X[n_tr:],
        y[n_tr:],
        n_splits=3,
        embargo_pct=0.02,
    )
    res_b = fit_calibrated_classifier_with_purged_cv(
        xgb_template,
        X[:n_tr],
        y[:n_tr],
        t0[:n_tr],
        t1[:n_tr],
        uniform,
        X[n_tr:],
        y[n_tr:],
        n_splits=3,
        embargo_pct=0.02,
    )

    pa = res_a.calibrated.predict_proba(X[n_tr:])
    pb = res_b.calibrated.predict_proba(X[n_tr:])
    assert np.max(np.abs(pa - pb)) > 1e-3, (
        "sample_weight did not influence the calibrated XGBoost — propagation broken"
    )
