"""Amihud Illiquidity — rolling OLS slope of |return| on dollar volume (audit V1).

Classical Amihud (2002) computes ``mean(|r_t| / V_t·P_t)`` element-wise per bar.
On bar data with near-zero volumes that produces inf/NaN outliers and unstable
features (AFML audit Vulnerability 1). We replace it with a **rolling OLS
slope** of ``|return|`` regressed on dollar volume over a causal window — the
same statistical content (price impact per unit of flow) but numerically
robust.

Strictly causal: ``.shift(1)`` on the rolling output.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from afml.features.kyle import MIN_VARIANCE, _rolling_ols_slope


def amihud_lambda(
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    *,
    window: int = 20,
) -> npt.NDArray[np.float64]:
    """Rolling OLS Amihud illiquidity slope.

    Slope of ``|log_return_t|`` regressed on ``dollar_volume_t`` over a causal
    window. Replaces the classical element-wise mean of ``|r| / $vol`` to avoid
    inf/NaN outliers when ``volume`` is near zero (AFML audit V1).

    Returns 1-D ``float64`` array of the same length as inputs.
    """
    if close.shape != volume.shape:
        raise ValueError("close and volume must have the same shape")
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")

    close_f = close.astype(np.float64, copy=False)
    volume_f = volume.astype(np.float64, copy=False)

    log_close = np.log(close_f)
    log_returns = np.empty_like(log_close)
    log_returns[0] = np.nan
    log_returns[1:] = np.diff(log_close)
    abs_returns = np.abs(log_returns)

    dollar_vol = close_f * volume_f

    raw = _rolling_ols_slope(dollar_vol, abs_returns, window, MIN_VARIANCE)
    return pd.Series(raw).shift(1).to_numpy(dtype=np.float64)
