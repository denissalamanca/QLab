"""Phase 2 — Brain 1: primary signals & Triple-Barrier labeling.

Public API:
- ``ewm_volatility`` — causal EWM rolling std of returns (used by CUSUM and
  Triple-Barrier).
- ``PrimaryAlpha``, ``register_alpha``, ``list_alpha_families``,
  ``get_alpha_class`` — plugin interface + registry.
- ``SymmetricCUSUM``, ``BollingerMeanReversion``, ``DonchianBreakout`` — the
  three locked plugin families (volatility events / ranges / momentum).
- ``apply_triple_barrier`` — Triple-Barrier labeling with dynamic EWM volatility.
- ``max_correlation``, ``is_orthogonal`` — orthogonality check vs deployed
  signal vectors in the Alpha Registry.
- ``Brain1Result``, ``run_brain1`` — orchestrator that runs each plugin, labels
  events, and logs experiments to the Alpha Registry.
"""

from afml.labeling.brain1 import Brain1Result, run_brain1
from afml.labeling.orthogonality import is_orthogonal, max_correlation
from afml.labeling.primary_alphas import (
    BollingerMeanReversion,
    DonchianBreakout,
    PrimaryAlpha,
    SymmetricCUSUM,
    get_alpha_class,
    list_alpha_families,
    register_alpha,
)
from afml.labeling.triple_barrier import TripleBarrierLabels, apply_triple_barrier
from afml.labeling.volatility import ewm_volatility

__all__ = [
    "BollingerMeanReversion",
    "Brain1Result",
    "DonchianBreakout",
    "PrimaryAlpha",
    "SymmetricCUSUM",
    "TripleBarrierLabels",
    "apply_triple_barrier",
    "ewm_volatility",
    "get_alpha_class",
    "is_orthogonal",
    "list_alpha_families",
    "max_correlation",
    "register_alpha",
    "run_brain1",
]
