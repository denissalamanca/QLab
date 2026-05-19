"""Chow-type Dickey-Fuller structural-break test (secondary, Blueprint §10.1).

GSADF is the primary regime-break detector; the Chow test is the secondary
confirmation. Where GSADF scans for *explosive* roots, the Chow test asks a
narrower question: did the autoregressive dynamics **break** at a candidate
point in time?

We fit a Dickey-Fuller regression ``Δy_t = α + β·y_{t-1} + ε_t`` on:

- the full sample (pooled), giving ``SSR_pooled``;
- the two sub-samples split at the breakpoint, giving ``SSR_1 + SSR_2``.

The Chow F-statistic compares the restricted (pooled) and unrestricted
(split) residual sums of squares:

::

    F = [ (SSR_pooled - (SSR_1 + SSR_2)) / k ]
        / [ (SSR_1 + SSR_2) / (T - 2k) ]

with ``k = 2`` parameters (intercept + AR term). Under the null of no break,
``F ~ F(k, T - 2k)``; a large F rejects parameter stability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import f as f_dist

DF_N_PARAMS: int = 2  # intercept + lagged level
MIN_SUBSAMPLE_POINTS: int = 5
DEFAULT_BREAKPOINT_FRAC: float = 0.5
DEFAULT_ALPHA: float = 0.05


def _df_ssr(y: npt.NDArray[np.float64]) -> float:
    """Residual sum of squares of the DF regression ``Δy = α + β·y_{t-1}``."""
    if y.size < MIN_SUBSAMPLE_POINTS:
        return float("nan")
    dy = np.diff(y)
    ylag = y[:-1]
    design = np.column_stack([np.ones_like(ylag), ylag])
    coef, _resid, _rank, _sv = np.linalg.lstsq(design, dy, rcond=None)
    residuals = dy - design @ coef
    return float(np.sum(residuals**2))


@dataclass(frozen=True, slots=True)
class ChowBreakResult:
    """Output of :func:`chow_break_test`.

    Attributes
    ----------
    f_statistic
        The Chow F-statistic at the tested breakpoint.
    critical_value
        ``F(k, T-2k)`` critical value at ``1 - alpha``.
    breakpoint_index
        Sample index where the break was tested.
    is_break
        ``True`` iff ``f_statistic > critical_value``.
    """

    f_statistic: float
    critical_value: float
    breakpoint_index: int
    is_break: bool


def chow_break_test(
    series: npt.NDArray[np.floating],
    *,
    breakpoint_frac: float = DEFAULT_BREAKPOINT_FRAC,
    alpha: float = DEFAULT_ALPHA,
) -> ChowBreakResult:
    """Chow F-test for a structural break in the DF regression.

    Parameters
    ----------
    series
        Price level series.
    breakpoint_frac
        Fraction of the sample at which to test for a break (default 0.5).
    alpha
        Right-tail significance level.

    Returns
    -------
    :class:`ChowBreakResult`.

    Raises
    ------
    ValueError
        If the series is too short to fit both sub-samples.
    """
    y = np.asarray(series, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"series must be 1-D, got shape {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("series must be finite")
    n = y.size
    bp = int(np.floor(breakpoint_frac * n))
    if bp < MIN_SUBSAMPLE_POINTS or (n - bp) < MIN_SUBSAMPLE_POINTS:
        raise ValueError(
            f"breakpoint {bp} leaves a sub-sample below {MIN_SUBSAMPLE_POINTS} points "
            f"(n={n}); choose a more central breakpoint_frac or a longer series"
        )

    ssr_pooled = _df_ssr(y)
    ssr_1 = _df_ssr(y[:bp])
    ssr_2 = _df_ssr(y[bp:])
    ssr_split = ssr_1 + ssr_2

    dof_denom = n - 2 * DF_N_PARAMS
    if dof_denom <= 0 or ssr_split <= 0.0:
        raise ValueError("insufficient degrees of freedom / degenerate residuals for Chow test")

    numerator = (ssr_pooled - ssr_split) / DF_N_PARAMS
    denominator = ssr_split / dof_denom
    f_stat = numerator / denominator
    crit = float(f_dist.ppf(1.0 - alpha, DF_N_PARAMS, dof_denom))
    return ChowBreakResult(
        f_statistic=float(f_stat),
        critical_value=crit,
        breakpoint_index=bp,
        is_break=f_stat > crit,
    )
