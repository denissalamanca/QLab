"""Indicator matrix, concurrency, and average-uniqueness sample weights.

AFML Ch. 4.4-4.5 / Blueprint §7.1. The mathematical chain:

- **Indicator matrix** ``I[t, i] ∈ {0, 1}``: 1 iff event ``i``'s label horizon
  ``[t0_i, t1_i]`` spans bar ``t``.
- **Concurrency** ``c_t = Σ_i I[t, i]``: number of label horizons simultaneously
  active at bar ``t``.
- **Uniqueness** ``u_{t, i} = I[t, i] / max(c_t, 1)``: per-bar uniqueness — the
  fraction of bar ``t``'s information that event ``i`` "owns".
- **Average uniqueness** ``ū_i = Σ_t u_{t, i} / Σ_t I[t, i]``: a single scalar
  per event in ``[0, 1]``. Events covering many concurrently-active bars get
  ``ū_i ≪ 1`` and are downweighted; isolated events get ``ū_i ≈ 1``.

The implementation works at the **event-index granularity** (rows = bar
indices, columns = event indices). We do not materialize the full bar grid —
only the unique union of event timestamps. This keeps memory ``O(n_events)``
even on multi-million-bar tick frames.

Numba-JIT primitives for the inner loops; the public API is plain numpy.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt


@numba.njit(cache=True)
def _build_indicator_matrix(
    t0: npt.NDArray[np.int64],
    t1: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Inner loop for :func:`indicator_matrix` — ``O(n_events × n_grid)``.

    Returns an ``(n_grid, n_events)`` int8-valued matrix (stored as int64 for
    numba simplicity; caller may cast).
    """
    n_grid = grid.shape[0]
    n_events = t0.shape[0]
    out = np.zeros((n_grid, n_events), dtype=np.int64)
    for ev_i in range(n_events):
        start = t0[ev_i]
        end = t1[ev_i]
        # Binary search would speed this up but numba's native np.searchsorted
        # works inside @njit only on certain numba versions. The linear scan
        # is fine for the ≤ 10^4 events Phase 5 sees in practice.
        for t_i in range(n_grid):
            ts = grid[t_i]
            if start <= ts <= end:
                out[t_i, ev_i] = 1
    return out


def indicator_matrix(
    t0: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    t1: npt.NDArray[np.int64] | npt.NDArray[np.floating],
    grid: npt.NDArray[np.int64] | npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.int64]:
    """Build the AFML indicator matrix.

    Parameters
    ----------
    t0, t1
        Event horizon bounds. Same length ``n_events``. Any monotonic numeric
        encoding (POSIX ms, ``np.datetime64`` cast to int64, plain integer
        position).
    grid
        Optional bar-time vector. When ``None`` the grid is the sorted union of
        ``t0`` and ``t1`` (the minimum sufficient grid for concurrency
        computation). Supplying an explicit bar grid is useful when downstream
        consumers need per-bar concurrency at a tick-bar timestamp set.

    Returns
    -------
    Indicator matrix of shape ``(n_grid, n_events)``, dtype ``int64``.
    """
    t0_arr = np.asarray(t0, dtype=np.int64)
    t1_arr = np.asarray(t1, dtype=np.int64)
    if t0_arr.shape != t1_arr.shape or t0_arr.ndim != 1:
        raise ValueError(
            f"t0 and t1 must be 1-D and same shape, got {t0_arr.shape} / {t1_arr.shape}"
        )
    if np.any(t1_arr < t0_arr):
        raise ValueError("every t1[i] must be ≥ t0[i]")

    if grid is None:
        grid_arr = np.unique(np.concatenate([t0_arr, t1_arr]))
    else:
        grid_arr = np.asarray(grid, dtype=np.int64)
        if grid_arr.ndim != 1:
            raise ValueError(f"grid must be 1-D, got shape {grid_arr.shape}")

    return _build_indicator_matrix(t0_arr, t1_arr, grid_arr)


def concurrency_count(indicator_mat: npt.NDArray[np.integer]) -> npt.NDArray[np.int64]:
    """Per-bar concurrency ``c_t = Σ_i I[t, i]``.

    Returns a ``(n_grid,)`` int64 vector.
    """
    return np.asarray(indicator_mat.sum(axis=1), dtype=np.int64)


def average_uniqueness(
    indicator_mat: npt.NDArray[np.integer],
    concurrency: npt.NDArray[np.integer] | None = None,
) -> npt.NDArray[np.float64]:
    """Per-event average uniqueness ``ū_i ∈ [0, 1]``.

    Parameters
    ----------
    indicator_mat
        ``(n_grid, n_events)`` output of :func:`indicator_matrix`.
    concurrency
        Optional precomputed concurrency vector. If ``None``, computed here.

    Returns
    -------
    ``(n_events,)`` float64 vector.

    Notes
    -----
    ``ū_i = mean(u_{t, i} | I[t, i] = 1)`` where ``u_{t, i} = I[t, i] / c_t``.
    Bars where ``I[t, i] = 0`` are excluded from the mean (they contribute zero
    in both numerator and denominator of the AFML formula).

    Events with no active bars (``Σ_t I[t, i] = 0``) get ``ū_i = 0`` — a
    degenerate sample that should be filtered out upstream, never weighted.
    """
    if concurrency is None:
        concurrency = concurrency_count(indicator_mat)
    ind_arr = np.asarray(indicator_mat, dtype=np.float64)
    c_arr = np.asarray(concurrency, dtype=np.float64)
    # Per-bar per-event uniqueness; 0 / 0 → 0.
    safe_c = np.where(c_arr > 0.0, c_arr, 1.0)
    uniqueness = ind_arr / safe_c[:, None]
    uniqueness = np.where(ind_arr > 0.0, uniqueness, 0.0)

    active_bars_per_event = ind_arr.sum(axis=0)
    safe_active = np.where(active_bars_per_event > 0.0, active_bars_per_event, 1.0)
    avg_u = uniqueness.sum(axis=0) / safe_active
    # Events with zero active bars → ū_i = 0.
    avg_u = np.where(active_bars_per_event > 0.0, avg_u, 0.0)
    return np.asarray(avg_u, dtype=np.float64)
