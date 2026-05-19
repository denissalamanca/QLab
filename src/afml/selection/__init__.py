"""Phase 4 — Feature Selection & Orthogonalization (Blueprint §6).

Pipeline:

1. **Distance metric** (``afml.selection.distance``):
   ``d_ij = √(0.5 · (1 − ρ_ij))`` over the feature matrix. The square-root
   transform of the Pearson correlation is a proper metric (López de Prado 2018
   Ch. 4): symmetric, zero-diagonal, triangle inequality satisfied, identical
   columns map to 0, perfectly anti-correlated columns map to 1.

2. **Ward agglomerative clustering + ONC** (``afml.selection.clustering``):
   ``AgglomerativeClustering(linkage="ward", metric="precomputed")`` over a
   sweep ``k ∈ [k_min, k_max]``. The Optimal Number of Clusters is the silhouette
   maximizer, with a **stable-plateau tie-break** so a single noisy peak cannot
   hijack the choice.

3. **Purged + Embargoed K-Fold CV** (``afml.selection.purged_kfold``): label
   horizons ``[t0, t1]`` overlapping the test window are purged from train; an
   additional embargo trims forward-leaking samples right after the test fold.
   AFML Ch. 7. Phase 5 reuses this splitter for Brain 2's Index-Intersection DoD.

4. **Clustered MDA** (``afml.selection.mda``): for each cluster, shuffle every
   feature in the cluster *simultaneously* on the OOS fold and measure the
   degradation in **negative log-loss** (NOT accuracy — Blueprint anti-shortcut
   rule). Statistical significance via a paired one-sided t-test across folds.

5. **SFI fallback** (``afml.selection.sfi``): if Clustered MDA fails to reduce
   the surviving feature count by ≥ 20 %, fall back to Single Feature Importance
   — train one classifier per feature, rank by NLL improvement over the
   no-feature baseline.

6. **Orchestrator** (``afml.selection.pipeline``): the public ``select_features``
   entry point.

Banned, by design — caught both at code review and via grep guard:
- ``sklearn.model_selection.KFold`` / ``TimeSeriesSplit`` (use Purged K-Fold).
- MDI / Gini ``feature_importances_`` (use Clustered MDA).
- ``accuracy_score`` / ``roc_auc_score`` as primary (use NLL / Brier).
"""

from afml.selection.clustering import ONCResult, cluster_features_onc
from afml.selection.distance import afml_distance_matrix
from afml.selection.mda import ClusteredMDAResult, clustered_mda
from afml.selection.pipeline import SelectionResult, select_features
from afml.selection.purged_kfold import (
    PurgedKFold,
    PurgedKFoldSklearn,
    PurgedWalkForwardCV,
)
from afml.selection.sfi import SFIResult, single_feature_importance

__all__ = [
    "ClusteredMDAResult",
    "ONCResult",
    "PurgedKFold",
    "PurgedKFoldSklearn",
    "PurgedWalkForwardCV",
    "SFIResult",
    "SelectionResult",
    "afml_distance_matrix",
    "cluster_features_onc",
    "clustered_mda",
    "select_features",
    "single_feature_importance",
]
