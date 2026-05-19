"""Order Flow Imbalance (OFI) — rolling signed-volume aggregate.

Sign of each bar's price change times the bar's traded volume; summed over a
rolling window. Positive OFI ⇒ net buying pressure; negative ⇒ net selling.

Strictly causal via ``.shift(1)`` on the rolling sum.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def ofi(
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    *,
    window: int = 20,
) -> npt.NDArray[np.float64]:
    """Rolling Order Flow Imbalance.

    Parameters
    ----------
    close, volume : 1-D arrays of equal length.
    window : look-back in bars.
    """
    if close.shape != volume.shape:
        raise ValueError("close and volume must have the same shape")
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")

    dp = pd.Series(close).diff()
    sign = np.sign(dp.fillna(0.0))
    signed_vol = sign * pd.Series(volume)
    rolling_sum = signed_vol.rolling(window=window, min_periods=window).sum()
    return rolling_sum.shift(1).to_numpy(dtype=np.float64)
