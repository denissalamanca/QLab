"""Hasbrouck Flow — rolling sign-weighted ``sqrt(volume)`` aggregate.

Hasbrouck-style trade-flow proxy: ``Σ sign(ΔP_t) · √V_t`` over a rolling
window. The sqrt damps the influence of outlier-volume bars relative to a raw
OFI sum. Strictly causal via ``.shift(1)``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def hasbrouck_flow(
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    *,
    window: int = 20,
) -> npt.NDArray[np.float64]:
    """Rolling Hasbrouck sign-weighted sqrt-volume sum."""
    if close.shape != volume.shape:
        raise ValueError("close and volume must have the same shape")
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")

    dp = pd.Series(close).diff()
    sign = np.sign(dp.fillna(0.0))
    flow = sign * np.sqrt(pd.Series(volume).clip(lower=0.0))
    rolling_sum = flow.rolling(window=window, min_periods=window).sum()
    return rolling_sum.shift(1).to_numpy(dtype=np.float64)
