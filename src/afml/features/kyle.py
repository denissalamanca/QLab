"""Kyle's Lambda — linear price-impact coefficient.

Regression slope of ``ΔP_t`` on signed volume ``sign(ΔP_t) · V_t`` over a
rolling window. Higher ``λ`` ⇒ more illiquid (larger price impact per unit of
order flow).

Computed as ``λ = Cov(ΔP, V_signed) / Var(V_signed)``, shifted ``.shift(1)`` for
strict causality.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def kyle_lambda(
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    *,
    window: int = 20,
) -> npt.NDArray[np.float64]:
    """Rolling Kyle's λ."""
    if close.shape != volume.shape:
        raise ValueError("close and volume must have the same shape")
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")

    dp = pd.Series(close).diff()
    sign = np.sign(dp.fillna(0.0))
    signed_vol = sign * pd.Series(volume)

    cov = dp.rolling(window=window, min_periods=window).cov(signed_vol)
    var = signed_vol.rolling(window=window, min_periods=window).var()
    lam = cov / var.replace(0.0, np.nan)
    return lam.shift(1).to_numpy(dtype=np.float64)
