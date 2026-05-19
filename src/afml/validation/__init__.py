"""Phase 6 — Rigorous Validation & CPCV (Blueprint §8).

Public API:

- **Combinatorially Purged Cross-Validation** (:mod:`afml.validation.cpcv`):
  ``CombinatoriallyPurgedKFold`` — generates ``C(N, k)`` purged + embargoed
  combinations; ``construct_oos_paths`` — assembles ``C(N-1, k-1)`` synthetic
  out-of-sample paths from per-combination predictions (López de Prado AFML
  Ch. 12).
- **PBO** (:mod:`afml.validation.pbo`): ``compute_pbo`` — Bailey & López de
  Prado 2014 Probability of Backtest Overfitting.
- **DSR** (:mod:`afml.validation.dsr`): ``expected_max_sharpe`` (parametric
  E[max SR] under multiple testing) + ``deflated_sharpe_ratio`` (Sharpe
  inflation-adjusted significance probability).
- **FWER** (:mod:`afml.validation.fwer`): Bonferroni + Holm-Bonferroni
  controls for the candidate p-value set.
- **Target shuffling** (:mod:`afml.validation.target_shuffling`):
  ``DataLeakageError`` raised when a model retains predictive power on
  randomised labels.
- **Orchestrator** (:mod:`afml.validation.pipeline`): ``validate_strategy``
  ties everything into a single Phase 6 gate.
"""

from afml.validation.cpcv import (
    CombinatoriallyPurgedKFold,
    CPCVFold,
    construct_oos_paths,
)
from afml.validation.dsr import (
    DSRResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from afml.validation.fwer import bonferroni_threshold, holm_bonferroni
from afml.validation.pbo import PBOResult, compute_pbo
from afml.validation.pipeline import ValidationResult, validate_strategy
from afml.validation.target_shuffling import (
    DataLeakageError,
    TargetShufflingResult,
    target_shuffling_test,
)

__all__ = [
    "CPCVFold",
    "CombinatoriallyPurgedKFold",
    "DSRResult",
    "DataLeakageError",
    "PBOResult",
    "TargetShufflingResult",
    "ValidationResult",
    "bonferroni_threshold",
    "compute_pbo",
    "construct_oos_paths",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "holm_bonferroni",
    "target_shuffling_test",
    "validate_strategy",
]
