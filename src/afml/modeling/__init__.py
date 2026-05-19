"""Phase 5 — Brain 2 / Meta-Labeling (Blueprint §7).

The meta-labeler predicts ``P(success | features)`` for every Brain 1 event,
correcting for two AFML pathologies that ordinary classifiers cannot handle:

1. **Non-IID labels.** Triple-Barrier label horizons overlap. A simple
   ``RandomForestClassifier`` bootstrap that ignores overlap will repeatedly
   resample the same information and overweight clustered events.
2. **Probability miscalibration.** Random forests are notoriously
   uncalibrated near the extremes. We need ``predict_proba`` to be a *true*
   probability — Phase 7 sizes positions as a function of it.

Public API:

- **Concurrency primitives** (:mod:`afml.modeling.concurrency`):
  ``indicator_matrix``, ``concurrency_count``, ``average_uniqueness``.
- **Sequential Bootstrap** (:mod:`afml.modeling.sequential_bootstrap`):
  AFML Snippet 4.3 — iteratively draws low-overlap samples with probability
  proportional to remaining average uniqueness.
- **SBRF** (:mod:`afml.modeling.sbrf`):
  ``SequentiallyBootstrappedRandomForest`` — sequentially-bootstrapped
  custom ensemble; per-tree sequential draws + ``sample_weight = ū_i``.
- **Calibration** (:mod:`afml.modeling.calibration`):
  ``fit_calibrated_classifier`` — fits ``CalibratedClassifierCV`` with
  isotonic + sigmoid (Platt) using a held-out calibration set, selects the
  variant that minimises Brier score on independent holdout.
- **Orchestrator** (:mod:`afml.modeling.pipeline`):
  ``train_brain_two`` → :class:`BrainTwoResult`.
"""

from afml.modeling.calibration import CalibrationResult, fit_calibrated_classifier
from afml.modeling.concurrency import (
    average_uniqueness,
    concurrency_count,
    indicator_matrix,
)
from afml.modeling.pipeline import BrainTwoResult, train_brain_two
from afml.modeling.sbrf import SequentiallyBootstrappedRandomForest
from afml.modeling.sequential_bootstrap import sequential_bootstrap

__all__ = [
    "BrainTwoResult",
    "CalibrationResult",
    "SequentiallyBootstrappedRandomForest",
    "average_uniqueness",
    "concurrency_count",
    "fit_calibrated_classifier",
    "indicator_matrix",
    "sequential_bootstrap",
    "train_brain_two",
]
