"""AFML correlation-distance metric (Blueprint §6.1, López de Prado 2018 Ch. 4).

The transformation ``d_ij = √(0.5 · (1 − ρ_ij))`` turns the Pearson correlation
into a proper distance metric — non-negative, symmetric, with a zero diagonal,
and satisfying the triangle inequality. Identical columns map to ``d = 0``;
perfectly anti-correlated columns map to ``d = 1``. This is the input to the
Ward agglomerative clustering step.

Why this and not ``1 − ρ`` or ``1 − |ρ|``? Both fail the triangle inequality
in general, which breaks Ward linkage's geometric interpretation. The
square-root variant is the canonical AFML choice.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

DISTANCE_MAX: float = 1.0


def afml_distance_matrix(
    feature_matrix: npt.NDArray[np.floating],
    *,
    min_std: float = 1e-12,
) -> npt.NDArray[np.float64]:
    """Return the AFML pairwise distance matrix over feature columns.

    Parameters
    ----------
    feature_matrix
        Shape ``(n_samples, n_features)``. Must be finite (no NaN / Inf).
        Each column is one feature; correlations are computed *across* columns.
    min_std
        Numerical floor on per-column standard deviation. Columns with
        ``std < min_std`` are treated as constants and assigned correlation 0
        with every other column (distance ``√0.5 ≈ 0.7071``). Without this
        guard, ``np.corrcoef`` emits ``RuntimeWarning`` and returns NaN.

    Returns
    -------
    Square symmetric ``(n_features, n_features)`` matrix, dtype ``float64``,
    zero diagonal, values in ``[0, 1]``.

    Raises
    ------
    ValueError
        - ``feature_matrix`` is not 2-D.
        - ``feature_matrix`` contains any non-finite element.
        - ``feature_matrix`` has fewer than 2 rows or 2 columns.

    Notes
    -----
    Properties enforced by tests (``tests/unit/phase4/test_distance.py``):
    1. Symmetric: ``D == D.T`` exactly.
    2. Zero diagonal: ``D[i, i] == 0``.
    3. Bounded: ``0 ≤ D ≤ 1``.
    4. ``d(x, x) = 0`` for any non-constant column.
    5. ``d(x, -x) = 1`` for perfectly anti-correlated columns.
    6. Constant column → finite distance (``√0.5``), never NaN.
    """
    arr = np.asarray(feature_matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"feature_matrix must be 2-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("feature_matrix contains NaN or Inf — Phase 3 must drop them first")
    n_samples, n_features = arr.shape
    if n_samples < 2 or n_features < 2:
        raise ValueError(
            f"need ≥ 2 samples and ≥ 2 features, got shape ({n_samples}, {n_features})"
        )

    # Identify constant columns so np.corrcoef doesn't emit NaN.
    col_std = arr.std(axis=0, ddof=1)
    constant_cols = col_std < min_std

    # Substitute constant columns with mean-centered noise *only* to make
    # corrcoef numerically stable; we overwrite the affected rows/cols below.
    work = arr.copy()
    if constant_cols.any():
        # Small perturbation — same draw each call (seeded), preserves determinism.
        rng = np.random.default_rng(0)
        perturb = rng.standard_normal((n_samples, int(constant_cols.sum())))
        work[:, constant_cols] = perturb

    rho = np.corrcoef(work, rowvar=False)
    # Clip floating-point drift to the legal interval.
    rho = np.clip(rho, -1.0, 1.0)
    distance = np.sqrt(0.5 * (1.0 - rho))

    # Force exact symmetry (corrcoef can drift by ~ULP).
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)

    # Overwrite distances involving constant columns with the maximal-uncertainty
    # value √0.5 — they carry no information, so we want them treated as
    # equidistant from everything else.
    if constant_cols.any():
        sqrt_half = float(np.sqrt(0.5))
        for idx in np.where(constant_cols)[0]:
            distance[idx, :] = sqrt_half
            distance[:, idx] = sqrt_half
            distance[idx, idx] = 0.0

    return np.asarray(distance, dtype=np.float64)
