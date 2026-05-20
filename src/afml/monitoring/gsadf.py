"""Generalized Supremum ADF — explosive-root bubble detection (PWY 2011).

Phillips, Wu & Yu (2011) "Explosive Behavior in the 1990s NASDAQ: When Did
Exuberance Escalate Asset Values?" introduced the Supremum ADF (SADF) and its
generalisation (GSADF) — recursive *right-tailed* unit-root tests. Where the
ordinary ADF asks "is there a unit root?", the right-tailed version asks "is
the autoregressive root **explosive** (> 1)?" — the statistical signature of a
price bubble.

- **ADF statistic** on a window: regress ``Δy_t = α + β·y_{t-1} + ε_t`` and
  take the t-stat of ``β``. ``β > 0`` (large positive t) ⇒ explosive.
- **GSADF**: the double supremum of the ADF statistic over *all* sub-windows
  ``[r1, r2]`` with ``r2 - r1 ≥`` a minimum window. This flexible window
  origin/endpoint makes GSADF far more powerful than SADF at detecting bubbles
  that occupy only part of the sample.

Critical values are not analytic — they depend on the sample length and the
minimum window. We estimate them by Monte-Carlo simulation under the null
(a driftless Gaussian random walk), exactly as PWY prescribe: simulate many
random walks, compute each one's GSADF, take the 95th percentile.

The inner ADF regression (lag 0, with intercept) is numba-JIT'd so the
``O(T²)`` window sweep — and the Monte-Carlo over hundreds of those sweeps —
runs in well under a second for the sample sizes Phase 8 monitors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
import numpy.typing as npt

# Minimum regression length: intercept + lag-1 term needs ≥ 3 points for a
# finite t-stat (dof = m - 2 ≥ 1).
MIN_REGRESSION_POINTS: int = 3
# AFML 0-8 final audit V2: an absolute floor on the GSADF minimum window
# (r0) so the inner OLS always has enough degrees of freedom. A window below
# this is statistically meaningless and risks degenerate regressions; series
# shorter than this floor return "no bubble" rather than attempting to fit.
MIN_WINDOW_OBSERVATIONS: int = 20
DEFAULT_MIN_WINDOW_FRAC: float = 0.2
DEFAULT_N_SIMULATIONS: int = 199
DEFAULT_QUANTILE: float = 0.95
_TINY: float = 1e-12
# AFML 0-9 final integration audit V5: a perfectly stagnant window (price
# flatlined for an illiquid session) makes every lagged difference zero, so the
# ADF design matrix is singular. Below this variance floor the level series
# carries no explosive-root evidence — return 0.0 instead of fitting a
# degenerate (singular) regression that would yield -inf / NaN / LinAlgError.
_MIN_VARIANCE: float = 1e-10


def _effective_min_window(n: int, min_window_frac: float) -> int:
    """Resolve the GSADF minimum window with the absolute floor enforced.

    ``r0 = max(MIN_WINDOW_OBSERVATIONS, ⌊min_window_frac · n⌋)`` — never below
    the audit floor, so the OLS degrees-of-freedom guarantee always holds.
    """
    return max(MIN_WINDOW_OBSERVATIONS, int(np.floor(min_window_frac * n)))


@numba.njit(cache=True)
def _adf_tstat(y: npt.NDArray[np.float64]) -> float:
    """ADF t-statistic (lag 0, intercept) for explosive-root detection.

    Regress ``Δy_t = α + β·y_{t-1} + ε_t`` via the analytic 2-variable OLS and
    return ``β̂ / SE(β̂)``. Returns ``-inf`` for degenerate windows so they
    never win the supremum.
    """
    n = y.shape[0]
    if n < MIN_REGRESSION_POINTS:
        return -np.inf
    m = n - 1
    # Δy and the lagged level.
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    for i in range(m):
        ylag = y[i]
        dy = y[i + 1] - y[i]
        sx += ylag
        sy += dy
        sxx += ylag * ylag
        sxy += ylag * dy
    denom = m * sxx - sx * sx
    if abs(denom) < _TINY:
        return -np.inf
    beta = (m * sxy - sx * sy) / denom
    alpha = (sy - beta * sx) / m
    ss = 0.0
    for i in range(m):
        resid = (y[i + 1] - y[i]) - alpha - beta * y[i]
        ss += resid * resid
    dof = m - 2
    if dof <= 0:
        return -np.inf
    sigma2 = ss / dof
    var_beta = sigma2 * m / denom
    if var_beta <= 0.0:
        return -np.inf
    return beta / np.sqrt(var_beta)


@numba.njit(cache=True)
def _gsadf_statistic(y: npt.NDArray[np.float64], min_window: int) -> float:
    """Double-sup ADF over every sub-window ``[r1, r2]`` with length ≥ min_window."""
    n = y.shape[0]
    best = -np.inf
    for r2 in range(min_window, n + 1):
        for r1 in range(0, r2 - min_window + 1):
            stat = _adf_tstat(y[r1:r2])
            best = max(best, stat)
    return best


def gsadf_statistic(
    series: npt.NDArray[np.floating],
    *,
    min_window_frac: float = DEFAULT_MIN_WINDOW_FRAC,
) -> float:
    """Compute the GSADF statistic of a price (level) series.

    Parameters
    ----------
    series
        Price level series (NOT returns — GSADF tests the level for an
        explosive autoregressive root).
    min_window_frac
        Minimum sub-window length as a fraction of the sample. PWY recommend
        ``r0 ≈ 0.01 + 1.8/√T``; ``0.2`` is a robust default for the
        few-hundred-bar windows Phase 8 monitors.

    Returns
    -------
    The GSADF statistic (a right-tailed ADF t-value supremum).
    """
    y = np.asarray(series, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"series must be 1-D, got shape {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("series must be finite")
    n = y.size
    # AFML 0-9 final audit V5: stagnant (zero-variance) series → no explosive
    # root. Guard before the window sweep so a flatlined market can't drive a
    # singular regression to -inf.
    if float(np.var(y)) < _MIN_VARIANCE:
        return 0.0
    min_window = _effective_min_window(n, min_window_frac)
    # AFML 0-8 final audit V2: too short for a meaningful regression — return
    # 0.0 (no explosive evidence) rather than attempting a degenerate OLS.
    if n < min_window:
        return 0.0
    stat = float(_gsadf_statistic(y, min_window))
    # Defensive: every sub-window was degenerate (-inf). Report no evidence
    # rather than leaking a non-finite statistic into events / serialization.
    return stat if np.isfinite(stat) else 0.0


def gsadf_critical_value(
    n: int,
    *,
    min_window_frac: float = DEFAULT_MIN_WINDOW_FRAC,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    quantile: float = DEFAULT_QUANTILE,
    random_state: int = 0,
) -> float:
    """Monte-Carlo critical value of GSADF under the random-walk null.

    Simulates ``n_simulations`` driftless Gaussian random walks of length
    ``n``, computes each one's GSADF, and returns the ``quantile`` of that
    null distribution (default 95th percentile → the 5% right-tail test).
    """
    if n < MIN_REGRESSION_POINTS:
        raise ValueError(f"n must be ≥ {MIN_REGRESSION_POINTS}, got {n}")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    min_window = _effective_min_window(n, min_window_frac)
    rng = np.random.default_rng(random_state)
    stats = np.empty(n_simulations, dtype=np.float64)
    for s in range(n_simulations):
        rw = np.cumsum(rng.standard_normal(n))
        stats[s] = _gsadf_statistic(rw, min_window)
    return float(np.quantile(stats, quantile))


@dataclass(frozen=True, slots=True)
class BubbleDetectionResult:
    """Output of :func:`detect_bubble`.

    Attributes
    ----------
    gsadf_statistic
        Observed GSADF on the input series.
    critical_value
        Monte-Carlo critical value under the random-walk null.
    is_bubble
        ``True`` iff ``gsadf_statistic > critical_value`` — the explosive-root
        rejection that fires a ``MARKET_REGIME_BREAK``.
    quantile
        The critical-value quantile used (e.g. 0.95).
    """

    gsadf_statistic: float
    critical_value: float
    is_bubble: bool
    quantile: float


def detect_bubble(
    series: npt.NDArray[np.floating],
    *,
    min_window_frac: float = DEFAULT_MIN_WINDOW_FRAC,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    quantile: float = DEFAULT_QUANTILE,
    random_state: int = 0,
) -> BubbleDetectionResult:
    """Run GSADF + Monte-Carlo critical value and decide bubble / no-bubble.

    This is the single entry point the Phase 8 monitor calls per asset.
    """
    y = np.asarray(series, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"series must be 1-D, got shape {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("series must be finite")
    n = y.size
    # AFML 0-9 final audit V5: a stagnant (zero-variance) window has no
    # explosive root and would otherwise produce a singular ADF regression.
    # Short-circuit to "no bubble" without touching the Monte-Carlo critical
    # value.
    if float(np.var(y)) < _MIN_VARIANCE:
        return BubbleDetectionResult(
            gsadf_statistic=0.0,
            critical_value=float("inf"),
            is_bubble=False,
            quantile=quantile,
        )
    min_window = _effective_min_window(n, min_window_frac)
    # AFML 0-8 final audit V2: a series shorter than the minimum window can't
    # support the GSADF regressions — return "no bubble" safely without
    # touching the (expensive, and here undefined) Monte-Carlo critical value.
    if n < min_window:
        return BubbleDetectionResult(
            gsadf_statistic=0.0,
            critical_value=float("inf"),
            is_bubble=False,
            quantile=quantile,
        )
    stat = gsadf_statistic(y, min_window_frac=min_window_frac)
    crit = gsadf_critical_value(
        n,
        min_window_frac=min_window_frac,
        n_simulations=n_simulations,
        quantile=quantile,
        random_state=random_state,
    )
    return BubbleDetectionResult(
        gsadf_statistic=stat,
        critical_value=crit,
        is_bubble=stat > crit,
        quantile=quantile,
    )
