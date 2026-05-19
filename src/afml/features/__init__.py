"""Phase 3 — microstructure feature engineering.

Public API:
- ``FeatureSpec`` + ``register_feature`` + ``list_features`` — feature registry.
- Per-feature implementations (``roll_measure``, ``corwin_schultz_spread``,
  ``ofi``, ``kyle_lambda``, ``amihud_lambda``, ``hasbrouck_flow``,
  ``shannon_entropy``, ``lempel_ziv_complexity``).
- ``compute_features`` — pipeline orchestrator. Returns a feature matrix
  sampled at event timestamps with ≥ 50 strictly-causal columns.
"""

from afml.features.amihud import amihud_lambda
from afml.features.base import (
    FeatureSpec,
    list_features,
    register_feature,
    reset_registry_for_tests,
)
from afml.features.corwin_schultz import corwin_schultz_spread
from afml.features.hasbrouck import hasbrouck_flow
from afml.features.kyle import kyle_lambda
from afml.features.lempel_ziv import lempel_ziv_complexity
from afml.features.ofi import ofi
from afml.features.pipeline import (
    DEFAULT_WINDOWS,
    StationarityRescueReport,
    compute_features,
    compute_features_with_report,
)
from afml.features.roll import roll_measure
from afml.features.shannon import shannon_entropy

__all__ = [
    "DEFAULT_WINDOWS",
    "FeatureSpec",
    "StationarityRescueReport",
    "amihud_lambda",
    "compute_features",
    "compute_features_with_report",
    "corwin_schultz_spread",
    "hasbrouck_flow",
    "kyle_lambda",
    "lempel_ziv_complexity",
    "list_features",
    "ofi",
    "register_feature",
    "reset_registry_for_tests",
    "roll_measure",
    "shannon_entropy",
]
