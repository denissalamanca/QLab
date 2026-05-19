"""Fixed-Width Fractional Differencing (Blueprint §3.2, AFML audit V2).

The standard ``.diff()`` operator (integer differencing) destroys market memory
to achieve stationarity. Fractional differencing preserves long-range dependence
by interpolating between identity (d = 0) and first-difference (d = 1).

This implementation enforces the **fixed-width** variant (López de Prado 2018,
Chapter 5): the convolution weights are truncated at the first index where
``|ω_l| < τ``, giving a hard upper bound ``l*`` on the look-back. Expanding-window
variants would silently grow the look-back as more history accrues — that
violates causality (see ``causality.py``).

**Constant-history invariant (AFML audit Vulnerability 2):**
``ffd_apply`` returns an array of length ``n - l*``. The first ``l*`` rows of
the raw fixed-window convolution are dropped so the resulting series consists
**only** of points that have a complete history of length ``l*``. This is what
keeps the marginal distribution stationary and the ADF test honest — every
output row is computed from exactly ``l*`` input lags, never fewer.

The numba-JIT'd inner loops handle multi-year tick datasets in seconds rather
than minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
import numpy.typing as npt

from afml.data.stationarity import adf_pvalue


@numba.njit(cache=True)
def _compute_weights(d: float, tol: float, max_window: int) -> npt.NDArray[np.float64]:
    """Compute FFD weights ω_0..ω_{l*-1} per Blueprint §3.2.

    ``ω_0 = 1``; ``ω_k = -ω_{k-1} · (d - k + 1) / k``. Stops at the first ``k``
    with ``|ω_k| < tol``. The window length ``l*`` equals the number of weights
    returned.
    """
    weights = np.empty(max_window, dtype=np.float64)
    weights[0] = 1.0
    last = 1
    for k in range(1, max_window):
        w = -weights[k - 1] * (d - k + 1.0) / k
        if abs(w) < tol:
            break
        weights[k] = w
        last = k + 1
    return weights[:last]


def ffd_weights(
    d: float, tol: float = 1e-5, *, max_window: int = 100_000
) -> npt.NDArray[np.float64]:
    """Public API for FFD weight computation.

    Parameters
    ----------
    d : fractional differencing order in (0, 1].
    tol : weight magnitude below which the window is truncated. The Blueprint
        fixes ``τ = 1e-5``.
    max_window : safety upper bound; the iteration always terminates before this
        for reasonable ``d``.

    Returns
    -------
    1-D ``float64`` array of length ``l*`` (the fixed window).
    """
    if not 0.0 < d <= 1.0:
        raise ValueError(f"d must be in (0, 1]; got {d}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive; got {tol}")
    return _compute_weights(d, tol, max_window)


@numba.njit(cache=True)
def _apply_fixed_window(
    series: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply fixed-width FFD weights to ``series``, returning length ``n - l*``.

    Only output points with a complete ``l*``-long history are emitted — the
    first ``l*`` rows of the raw convolution are dropped (AFML audit V2). This
    is what makes every output row identically distributed under stationarity.
    """
    n = series.shape[0]
    n_w = weights.shape[0]
    if n <= n_w:
        return np.empty(0, dtype=np.float64)
    # Output index 0 corresponds to input index n_w (which uses inputs
    # [1, n_w]). Output index i corresponds to input index n_w + i, using
    # inputs [i + 1, n_w + i] — always exactly ``n_w`` lags.
    out_len = n - n_w
    out = np.empty(out_len, dtype=np.float64)
    for out_i in range(out_len):
        center = n_w + out_i
        s = 0.0
        for k in range(n_w):
            s += weights[k] * series[center - k]
        out[out_i] = s
    return out


def ffd_apply(
    series: npt.NDArray[np.float64],
    d: float,
    *,
    tol: float = 1e-5,
) -> npt.NDArray[np.float64]:
    """Apply FFD with order ``d`` and tolerance ``tol`` to a 1-D price series.

    Returns an array of length ``n - l*``, where ``l* = len(ffd_weights(d, tol))``
    is the fixed window. The first ``l*`` rows of the convolution — which would
    have partial or boundary-effect history — are strictly dropped (AFML audit
    Vulnerability 2). Every emitted row is computed from exactly ``l*`` lags
    of the input, guaranteeing the marginal distribution is stationary by
    construction.
    """
    if series.ndim != 1:
        raise ValueError(f"series must be 1-D; got shape {series.shape}")
    s = series.astype(np.float64, copy=False)
    w = ffd_weights(d, tol)
    return _apply_fixed_window(s, w)


# ---------------------------------------------------------------------------------
# Optimal-d search.
# ---------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FFDResult:
    """Outcome of the optimal-``d`` search.

    Attributes
    ----------
    d_optimal : the lowest ``d`` for which the FFD-transformed series passes
        ADF at the given threshold; ``None`` if no candidate in the grid passed.
    window_length : ``l*`` (number of FFD weights) at ``d_optimal``.
    series : the FFD-transformed series at ``d_optimal``.
    adf_pvalue_at_optimum : the ADF p-value at the chosen ``d``.
    sweep : per-``d`` ADF p-value mapping (for the parameter-sweep log mandated
        by PRD §4 DoD).
    """

    d_optimal: float | None
    window_length: int | None
    series: npt.NDArray[np.float64] | None
    adf_pvalue_at_optimum: float | None
    sweep: dict[float, float]


def find_optimal_d(
    series: npt.NDArray[np.float64],
    *,
    d_grid: npt.NDArray[np.float64] | None = None,
    adf_p_threshold: float = 0.05,
    tol: float = 1e-5,
) -> FFDResult:
    """Find the smallest ``d`` such that the FFD-transformed series is stationary.

    Smallest ``d`` ⇒ smallest transformation ⇒ maximum memory preserved (lowest
    ``d`` is equivalent to maximum Pearson correlation with the original series
    among the candidates that pass ADF — see PRD §4).

    Anti-Lazy: the full sweep is logged in ``FFDResult.sweep`` so the choice is
    auditable and not a hidden curve-fit.

    Parameters
    ----------
    series : 1-D price (or log-price) series.
    d_grid : candidates to test. Defaults to ``linspace(0.05, 1.0, 20)``.
    adf_p_threshold : ADF p-value below which a series is considered stationary.
    tol : FFD weight tolerance (fixes ``l*``).
    """
    if d_grid is None:
        d_grid = np.linspace(0.05, 1.0, 20)

    sweep: dict[float, float] = {}
    best_d: float | None = None
    best_series: npt.NDArray[np.float64] | None = None
    best_p: float | None = None
    best_l: int | None = None

    for d in d_grid:
        d_f = float(d)
        ffd = ffd_apply(series, d_f, tol=tol)
        if ffd.size < 20:
            # ADF demands ≥ 20 obs; large d / huge window leaves too few.
            sweep[d_f] = float("nan")
            continue
        try:
            p = adf_pvalue(ffd)
        except ValueError:
            sweep[d_f] = float("nan")
            continue
        sweep[d_f] = p
        if p < adf_p_threshold and best_d is None:
            # First (smallest) d to pass — that's our optimum.
            best_d = d_f
            best_series = ffd
            best_p = p
            best_l = ffd_weights(d_f, tol).shape[0]
            # Continue the sweep to populate the audit log, but don't update best.

    return FFDResult(
        d_optimal=best_d,
        window_length=best_l,
        series=best_series,
        adf_pvalue_at_optimum=best_p,
        sweep=sweep,
    )
