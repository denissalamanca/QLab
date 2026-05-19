"""Kyle's Lambda — linear price-impact coefficient via rolling OLS (AFML audit V1).

Definition (Kyle 1985): ``λ`` is the regression slope of price change on signed
volume, ``ΔP ~ λ · V_signed``. Implemented here as a **rolling OLS slope** over a
causal window of bars, NOT as an instantaneous element-wise ``ΔP / V`` ratio —
the latter explodes whenever volume is near zero and produces inf/NaN outliers.

The numba-JIT'd inner loop computes the slope as
``Σ (x_i - x̄)(y_i - ȳ) / Σ (x_i - x̄)²`` over each window. When the variance of
the regressor falls below ``MIN_VARIANCE`` (numerical ill-conditioning), the
slope is set to NaN — Phase 4 ONC will see a missing observation rather than an
infinite outlier.

Strictly causal via ``.shift(1)`` (the audit V4 directive applied to features).
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd

# Variance floor below which the OLS slope is numerically meaningless.
# Tuned to be well above float64 underflow but below any realistic signed-volume
# variance on real ticks.
MIN_VARIANCE: float = 1e-12


@numba.njit(cache=True)
def _rolling_ols_slope(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    window: int,
    min_variance: float,
) -> npt.NDArray[np.float64]:
    """Causal rolling OLS slope of ``y`` regressed on ``x`` over ``window`` bars.

    Output at index ``t`` uses ``[t-window+1, t]``. Callers must additionally
    ``.shift(1)`` if strict pre-event causality is required.

    Returns NaN where any input is NaN or where ``Var(x) < min_variance``.
    """
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2 or window > n:
        return out

    for t in range(window - 1, n):
        # Two-pass mean / variance to avoid catastrophic cancellation.
        sx = 0.0
        sy = 0.0
        for i in range(t - window + 1, t + 1):
            xi = x[i]
            yi = y[i]
            if np.isnan(xi) or np.isnan(yi):
                sx = np.nan
                break
            sx += xi
            sy += yi
        if np.isnan(sx):
            continue
        mx = sx / window
        my = sy / window

        cov_xy = 0.0
        var_x = 0.0
        for i in range(t - window + 1, t + 1):
            dx = x[i] - mx
            dy = y[i] - my
            cov_xy += dx * dy
            var_x += dx * dx
        if var_x < min_variance * window:
            continue
        out[t] = cov_xy / var_x
    return out


def kyle_lambda(
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    *,
    window: int = 20,
) -> npt.NDArray[np.float64]:
    """Rolling OLS Kyle's λ over a causal window.

    Parameters
    ----------
    close, volume : 1-D arrays of equal length (bar close prices and volumes).
    window : look-back window in bars for the rolling regression.

    Returns
    -------
    1-D ``float64`` array of the same length as inputs; NaN warm-up at the head
    (window - 1 entries) plus an additional ``.shift(1)`` for strict causality.
    Output positions where ``Var(signed_volume) < MIN_VARIANCE`` are NaN.
    """
    if close.shape != volume.shape:
        raise ValueError("close and volume must have the same shape")
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")

    close_f = close.astype(np.float64, copy=False)
    volume_f = volume.astype(np.float64, copy=False)

    dp = np.diff(close_f, prepend=close_f[0])
    sign = np.sign(dp)
    signed_vol = sign * volume_f

    raw = _rolling_ols_slope(signed_vol, dp, window, MIN_VARIANCE)
    return pd.Series(raw).shift(1).to_numpy(dtype=np.float64)
