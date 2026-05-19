"""Concept-drift monitoring via SHAP feature-importance rank stability (§10.2).

A model can keep predicting confidently while the *reasons* behind its
predictions silently shift — the market regime changes, and features that once
mattered stop mattering. AFML's Phase 8 detector watches the **rank order** of
SHAP feature importances:

1. At training time, record the per-feature mean ``|SHAP|`` importance vector.
2. Periodically on live OOS data, recompute the same vector.
3. Take the **Spearman rank correlation** between training and live importances.
4. If it drops below ``0.5``, the model is explaining its outputs with a
   materially different feature ranking → emit ``CONCEPT_DRIFT_ALERT`` and halt
   Agent 7.

Spearman (rank) correlation — not Pearson — because we care about the *order*
of feature importance, not its absolute scale (SHAP magnitudes drift with
volatility even absent concept drift).

``compute_shap_importance`` uses ``shap.TreeExplainer`` for tree ensembles
(SBRF / XGBoost / RandomForest). The drift decision itself operates on plain
importance vectors, so it is fully testable without invoking SHAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr

DEFAULT_DRIFT_THRESHOLD: float = 0.5


def compute_shap_importance(
    model: Any,
    X: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """Per-feature mean ``|SHAP value|`` importance for a tree model.

    Parameters
    ----------
    model
        A fitted tree-based classifier (RandomForest / XGBoost / SBRF's
        underlying trees) compatible with ``shap.TreeExplainer``.
    X
        ``(n_samples, n_features)`` data to attribute over (training data for
        the baseline, live OOS data for the current reading).

    Returns
    -------
    ``(n_features,)`` vector of mean absolute SHAP values per feature.
    """
    import shap  # noqa: PLC0415 — heavy optional dep; import only when used

    X_arr = np.asarray(X, dtype=np.float64)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_arr)
    # Binary classifiers may return a list [class0, class1] or a 3-D array;
    # collapse to the positive-class attribution.
    arr = np.asarray(shap_values)
    n_dims_with_class_axis = 3  # (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[-1])
    elif arr.ndim == n_dims_with_class_axis:
        arr = arr[..., -1]
    return np.asarray(np.mean(np.abs(arr), axis=0), dtype=np.float64)


def spearman_rank_correlation(
    importance_a: npt.NDArray[np.floating],
    importance_b: npt.NDArray[np.floating],
) -> float:
    """Spearman rank correlation between two importance vectors.

    Returns ``1.0`` for identical orderings, ``-1.0`` for fully reversed.
    A constant vector (no rank information) yields ``0.0`` rather than NaN.
    """
    a = np.asarray(importance_a, dtype=np.float64)
    b = np.asarray(importance_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(
            f"importance vectors must be 1-D and same shape; got {a.shape} / {b.shape}"
        )
    if a.size < 2:
        raise ValueError("need ≥ 2 features to rank-correlate")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    rho, _pvalue = spearmanr(a, b)
    return float(rho)


@dataclass(frozen=True, slots=True)
class ConceptDriftResult:
    """Output of :func:`detect_concept_drift`.

    Attributes
    ----------
    spearman_rank_corr
        Rank correlation of training vs live SHAP importance.
    threshold
        The drift threshold (default 0.5).
    drifted
        ``True`` iff ``spearman_rank_corr < threshold`` — the
        ``CONCEPT_DRIFT_ALERT`` condition.
    """

    spearman_rank_corr: float
    threshold: float
    drifted: bool


def detect_concept_drift(
    training_importance: npt.NDArray[np.floating],
    live_importance: npt.NDArray[np.floating],
    *,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> ConceptDriftResult:
    """Decide whether feature-importance ranking has drifted.

    Blueprint §10.2: ``rank_corr < 0.5 ⇒ CONCEPT_DRIFT_ALERT``.
    """
    rho = spearman_rank_correlation(training_importance, live_importance)
    return ConceptDriftResult(
        spearman_rank_corr=rho,
        threshold=threshold,
        drifted=rho < threshold,
    )
