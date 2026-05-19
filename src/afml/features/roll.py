"""Roll Measure — effective-spread estimator from serial covariance of price changes.

Roll (1984): ``R = 2 · sqrt(max(0, -Cov(ΔP_t, ΔP_{t-1})))``.

Computed on a rolling window with ``.shift(1)`` so the value at index ``t``
depends only on price changes at indices ``< t``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def roll_measure(close: npt.NDArray[np.float64], *, window: int = 20) -> npt.NDArray[np.float64]:
    """Rolling Roll spread estimate.

    Parameters
    ----------
    close : 1-D array of bar close prices.
    window : look-back window for the rolling covariance (in bars).

    Returns
    -------
    1-D ``float64`` array of the same length. First ``window+1`` rows are NaN.
    """
    if close.ndim != 1:
        raise ValueError(f"close must be 1-D; got shape {close.shape}")
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")

    dp = pd.Series(close).diff()
    dp_lag = dp.shift(1)
    cov = dp.rolling(window=window, min_periods=window).cov(dp_lag)
    cov_causal = cov.shift(1)
    neg = (-cov_causal).clip(lower=0.0)
    return (2.0 * np.sqrt(neg)).to_numpy(dtype=np.float64)
