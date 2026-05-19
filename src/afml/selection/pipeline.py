"""Phase 4 orchestrator — full feature-selection pipeline (Blueprint §6).

Public entry point: :func:`select_features`. Given a Phase-3 feature matrix and
Brain-1 / Triple-Barrier labels with their ``[t0, t1]`` event horizons, returns
a :class:`SelectionResult` carrying:

* the AFML distance matrix,
* ONC cluster labels and silhouette curve,
* Clustered MDA per-cluster importances + p-values,
* the surviving (orthogonal, predictive) feature set,
* whether the SFI fallback was triggered (Clustered MDA failed to prune
  ≥ ``min_reduction_pct`` of the original features).

The pipeline does NOT mutate its inputs. ``X`` is converted to numpy float64
internally; the original DataFrame is kept untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from afml.selection.clustering import ONCResult, cluster_features_onc
from afml.selection.distance import afml_distance_matrix
from afml.selection.mda import (
    DEFAULT_SIGNIFICANCE_THRESHOLD,
    ClusteredMDAResult,
    clustered_mda,
)
from afml.selection.sfi import SFIResult, single_feature_importance

DEFAULT_MIN_REDUCTION_PCT: float = 0.20


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Output of :func:`select_features`.

    Attributes
    ----------
    feature_names
        Original feature names in matrix column order.
    distance_matrix
        ``(n_features, n_features)`` AFML distance matrix.
    onc
        :class:`ONCResult` from Ward + silhouette ONC.
    mda
        :class:`ClusteredMDAResult` from clustered permutation importance.
    sfi
        Single-feature-importance result, populated only if MDA failed to prune
        ≥ ``min_reduction_pct`` of the original features. Otherwise ``None``.
    surviving_features
        Names of the features kept by the final pipeline. Subset of
        ``feature_names``.
    used_sfi_fallback
        Whether the SFI fallback path was taken.
    cluster_to_features
        Convenience map ``cluster_id → [feature_name, ...]``.
    """

    feature_names: list[str]
    distance_matrix: npt.NDArray[np.float64]
    onc: ONCResult
    mda: ClusteredMDAResult
    sfi: SFIResult | None
    surviving_features: list[str]
    used_sfi_fallback: bool
    cluster_to_features: dict[int, list[str]]


def select_features(
    X: pl.DataFrame,
    y: npt.NDArray[np.integer],
    t0: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    t1: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    *,
    feature_columns: list[str] | None = None,
    timestamp_col: str = "timestamp",
    significance_threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD,
    min_reduction_pct: float = DEFAULT_MIN_REDUCTION_PCT,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    random_state: int = 0,
) -> SelectionResult:
    """Run the full Phase 4 selection pipeline.

    Parameters
    ----------
    X
        Phase-3 feature matrix as a :class:`polars.DataFrame`. May contain a
        timestamp column (``timestamp_col``) which will be excluded from the
        feature set automatically.
    y
        ``(n_samples,)`` binary labels from Phase 2's Triple-Barrier output,
        aligned row-for-row with ``X``.
    t0, t1
        ``(n_samples,)`` event-horizon bounds (see :class:`PurgedKFold`).
    feature_columns
        Explicit subset of feature column names. ``None`` ⇒ every column except
        ``timestamp_col``.
    timestamp_col
        Name of the timestamp column in ``X`` (if any) to exclude.
    significance_threshold
        Forwarded to :func:`clustered_mda`.
    min_reduction_pct
        Required relative dimensionality reduction. If Clustered MDA fails to
        shrink the surviving feature count by at least this fraction, the SFI
        fallback is invoked and its survivors override the MDA survivors.
    n_splits, embargo_pct
        Purged K-Fold knobs.
    random_state
        Seeds RandomForest + permutation RNG for reproducibility.

    Returns
    -------
    :class:`SelectionResult`.
    """
    if feature_columns is None:
        feature_columns = [c for c in X.columns if c != timestamp_col]
    if len(feature_columns) < 2:
        raise ValueError(f"need ≥ 2 feature columns, got {len(feature_columns)}")

    feature_matrix = X.select(feature_columns).to_numpy().astype(np.float64)
    n_samples, n_features = feature_matrix.shape
    if y.shape != (n_samples,):
        raise ValueError(f"y shape {y.shape} != (n_samples,) {(n_samples,)}")

    # 1. Distance matrix.
    distance_matrix = afml_distance_matrix(feature_matrix)

    # 2. Ward + ONC.
    onc = cluster_features_onc(distance_matrix)

    cluster_to_features: dict[int, list[str]] = {}
    for col_idx, cluster_id in enumerate(onc.labels.tolist()):
        cluster_to_features.setdefault(int(cluster_id), []).append(feature_columns[col_idx])

    # 3. Clustered MDA.
    mda = clustered_mda(
        feature_matrix,
        y,
        onc.labels,
        t0,
        t1,
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        significance_threshold=significance_threshold,
        random_state=random_state,
    )

    # Features surviving MDA = features in any kept cluster.
    mda_survivors: list[str] = []
    for cluster_id in mda.kept_clusters:
        mda_survivors.extend(cluster_to_features[cluster_id])

    # 4. SFI fallback gate.
    reduction = 1.0 - (len(mda_survivors) / n_features) if n_features > 0 else 0.0
    sfi: SFIResult | None = None
    used_sfi_fallback = False

    if reduction < min_reduction_pct:
        sfi = single_feature_importance(
            feature_matrix,
            y,
            t0,
            t1,
            n_splits=n_splits,
            embargo_pct=embargo_pct,
            random_state=random_state,
        )
        sfi_survivors = [feature_columns[j] for j in sfi.surviving_columns]
        # Take the intersection if MDA produced any survivors; otherwise SFI
        # alone — but ALWAYS enforce the ≥ min_reduction_pct constraint, even
        # if that means falling back to the top-k SFI features.
        if mda_survivors:
            union_set = set(mda_survivors) & set(sfi_survivors)
            surviving_features = [f for f in feature_columns if f in union_set]
        else:
            surviving_features = sfi_survivors

        # If even after SFI we still haven't reduced enough, hard-cap at top-k.
        target_count = int(np.floor(n_features * (1.0 - min_reduction_pct)))
        if len(surviving_features) > target_count and sfi is not None:
            sorted_by_imp = sorted(
                sfi.feature_importance.items(), key=lambda kv: kv[1], reverse=True
            )
            top_k_idx = {j for j, _ in sorted_by_imp[:target_count]}
            surviving_features = [feature_columns[j] for j in range(n_features) if j in top_k_idx]
        used_sfi_fallback = True
    else:
        surviving_features = mda_survivors

    return SelectionResult(
        feature_names=feature_columns,
        distance_matrix=distance_matrix,
        onc=onc,
        mda=mda,
        sfi=sfi,
        surviving_features=surviving_features,
        used_sfi_fallback=used_sfi_fallback,
        cluster_to_features=cluster_to_features,
    )
