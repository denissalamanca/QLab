"""Amihud's Lambda (illiquidity) — ``|return| / dollar_volume`` averaged over a window.

Higher ⇒ thinner book (a small flow moves prices a lot). Strictly causal: at
index ``t`` the value uses bar returns and dollar volumes at indices ``< t``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def amihud_lambda(
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    *,
    window: int = 20,
) -> npt.NDArray[np.float64]:
    """Rolling Amihud illiquidity measure."""
    if close.shape != volume.shape:
        raise ValueError("close and volume must have the same shape")
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")

    rets = pd.Series(close).pct_change().abs()
    dollar_vol = pd.Series(close) * pd.Series(volume)
    illiq = rets / dollar_vol.replace(0.0, np.nan)
    rolling_mean = illiq.rolling(window=window, min_periods=window).mean()
    return rolling_mean.shift(1).to_numpy(dtype=np.float64)
