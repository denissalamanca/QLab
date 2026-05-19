"""Phase 8 — MLOps & Structural Breaks (Blueprint §10).

Public API:

- **GSADF** (:mod:`afml.monitoring.gsadf`): ``detect_bubble`` /
  ``gsadf_statistic`` / ``gsadf_critical_value`` — Phillips-Wu-Yu 2011
  explosive-root bubble detection with Monte-Carlo critical values.
- **Chow** (:mod:`afml.monitoring.chow`): ``chow_break_test`` — secondary
  structural-break F-test on the Dickey-Fuller regression.
- **SHAP drift** (:mod:`afml.monitoring.shap_drift`):
  ``compute_shap_importance``, ``spearman_rank_correlation``,
  ``detect_concept_drift``.
- **Monitor** (:mod:`afml.monitoring.pipeline`): ``StructuralBreakMonitor`` —
  produces ``MarketRegimeBreak`` / ``ConceptDriftAlert`` events for Agent 8.
"""

from afml.monitoring.chow import ChowBreakResult, chow_break_test
from afml.monitoring.gsadf import (
    BubbleDetectionResult,
    detect_bubble,
    gsadf_critical_value,
    gsadf_statistic,
)
from afml.monitoring.pipeline import (
    DriftCheck,
    RegimeCheck,
    StructuralBreakMonitor,
)
from afml.monitoring.shap_drift import (
    ConceptDriftResult,
    compute_shap_importance,
    detect_concept_drift,
    spearman_rank_correlation,
)

__all__ = [
    "BubbleDetectionResult",
    "ChowBreakResult",
    "ConceptDriftResult",
    "DriftCheck",
    "RegimeCheck",
    "StructuralBreakMonitor",
    "chow_break_test",
    "compute_shap_importance",
    "detect_bubble",
    "detect_concept_drift",
    "gsadf_critical_value",
    "gsadf_statistic",
    "spearman_rank_correlation",
]
