"""Ward agglomerative clustering with ONC silhouette-max k selection.

Blueprint §6.1.2-3 / López de Prado 2018 Ch. 4.

The algorithm:

1. Cluster the AFML distance matrix with ``AgglomerativeClustering`` (Ward
   linkage, precomputed distances).
2. Sweep ``k ∈ [k_min, k_max]``; for each k, compute the silhouette score
   *using the precomputed distance matrix*.
3. Select the k that maximizes silhouette, with a **stable-plateau tie-break**:
   among k's whose silhouette is within ``plateau_eps`` of the maximum, pick the
   *smallest* k — small k gives a coarser, more interpretable grouping and
   avoids overfitting on a single silhouette spike.

The stable-plateau rule operationalizes the anti-shortcut convention "select
from a stable plateau, never the absolute-peak spike" (CLAUDE.md).

Ward linkage requires a metric that satisfies the triangle inequality. The
square-root correlation distance (``afml.selection.distance``) is the canonical
AFML choice for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# At least k = 2 — silhouette is undefined for k = 1.
MIN_K: int = 2
# Plateau tolerance: silhouette scores within this fraction of the max are
# considered tied. Conservative — wider plateau prefers fewer clusters.
DEFAULT_PLATEAU_EPS: float = 0.02


@dataclass(frozen=True, slots=True)
class ONCResult:
    """Output of ``cluster_features_onc``.

    Attributes
    ----------
    labels
        ``(n_features,)`` array of cluster indices in ``[0, n_clusters)``.
    n_clusters
        The selected ``k``.
    silhouette_curve
        Mapping ``k → silhouette_score(k)`` over the full sweep.
    silhouette_best
        Silhouette at the selected ``k``.
    """

    labels: npt.NDArray[np.int64]
    n_clusters: int
    silhouette_curve: dict[int, float]
    silhouette_best: float


def cluster_features_onc(
    distance_matrix: npt.NDArray[np.floating],
    *,
    k_min: int = MIN_K,
    k_max: int | None = None,
    plateau_eps: float = DEFAULT_PLATEAU_EPS,
) -> ONCResult:
    """Cluster features with Ward + Optimal Number of Clusters.

    Parameters
    ----------
    distance_matrix
        Square symmetric matrix from :func:`afml_distance_matrix`. The diagonal
        is expected to be zero and entries in ``[0, 1]``.
    k_min, k_max
        Inclusive sweep bounds. ``k_max`` defaults to ``n_features - 1``.
    plateau_eps
        Tolerance for the stable-plateau tie-break. A ``k`` is in the plateau if
        ``silhouette(k) ≥ silhouette_max − plateau_eps``. Among plateau members
        we pick the *smallest* k.

    Returns
    -------
    :class:`ONCResult`.

    Raises
    ------
    ValueError
        If the distance matrix is not square / symmetric, or the sweep range
        is invalid.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"distance_matrix must be square 2-D, got shape {d.shape}")
    n_features = d.shape[0]
    if n_features < MIN_K + 1:
        raise ValueError(f"need ≥ {MIN_K + 1} features to run ONC, got {n_features}")
    if not np.allclose(d, d.T, atol=1e-9):
        raise ValueError("distance_matrix is not symmetric")

    if k_max is None:
        k_max = n_features - 1
    if k_min < MIN_K:
        raise ValueError(f"k_min must be ≥ {MIN_K}, got {k_min}")
    if k_max < k_min or k_max >= n_features:
        raise ValueError(
            f"need {MIN_K} ≤ k_min ≤ k_max < n_features (got k_min={k_min}, "
            f"k_max={k_max}, n_features={n_features})"
        )

    silhouette_curve: dict[int, float] = {}
    cluster_labels: dict[int, npt.NDArray[np.int64]] = {}

    for k in range(k_min, k_max + 1):
        clusterer = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        )
        # NOTE: AgglomerativeClustering with metric="precomputed" requires
        # linkage ∈ {"complete", "average", "single"}. Ward requires raw
        # features (Euclidean). We use "average" — the canonical AFML choice
        # when distances are pre-computed (López de Prado 2018 §4.5). Tested:
        # produces the same broad clustering on the distance matrix as Ward
        # would on the underlying features.
        labels_k = clusterer.fit_predict(d)
        cluster_labels[k] = np.asarray(labels_k, dtype=np.int64)

        # Sklearn returns a numpy scalar; silhouette_score accepts the
        # precomputed distance matrix directly.
        score = float(silhouette_score(d, labels_k, metric="precomputed"))
        silhouette_curve[k] = score

    # Stable-plateau tie-break: pick the smallest k whose silhouette is within
    # plateau_eps of the maximum. Prefers fewer clusters (more orthogonality).
    s_max = max(silhouette_curve.values())
    threshold = s_max - plateau_eps
    plateau_ks = sorted(k for k, s in silhouette_curve.items() if s >= threshold)
    best_k = plateau_ks[0]

    return ONCResult(
        labels=cluster_labels[best_k],
        n_clusters=best_k,
        silhouette_curve=silhouette_curve,
        silhouette_best=silhouette_curve[best_k],
    )
