"""Stationarity + normality test wrappers.

Thin façades over ``statsmodels`` and ``scipy`` with the AFML-canonical
configuration baked in. Centralized so future phases use the same conventions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats
from statsmodels.tsa.stattools import adfuller


def adf_pvalue(
    series: npt.NDArray[np.float64],
    *,
    autolag: str = "AIC",
    regression: str = "c",
    maxlag: int | None = None,
) -> float:
    """Augmented Dickey-Fuller p-value.

    Strips NaNs (which arise from FFD's warm-up window) before testing.
    Returns the p-value; the caller compares it against the 0.05 threshold.
    """
    clean = series[~np.isnan(series)]
    if clean.size < 20:
        raise ValueError(f"ADF requires at least 20 non-NaN observations; got {clean.size}")
    # adfuller returns a 6-tuple: (stat, pvalue, usedlag, nobs, crit_values, icbest)
    result = adfuller(clean, autolag=autolag, regression=regression, maxlag=maxlag)
    return float(result[1])


def jarque_bera_statistic(returns: npt.NDArray[np.float64]) -> float:
    """Jarque-Bera test statistic on a returns series.

    Higher value ⇒ farther from normality. The information-bar selector picks
    the bar type that MINIMIZES this statistic (closest to normal returns).

    Strips NaN/infinite entries before testing.
    """
    clean = returns[np.isfinite(returns)]
    if clean.size < 10:
        raise ValueError(f"Jarque-Bera requires at least 10 finite observations; got {clean.size}")
    result = stats.jarque_bera(clean)
    return float(result.statistic)
