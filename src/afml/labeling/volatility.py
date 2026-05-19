"""EWM rolling-volatility primitive (shared by CUSUM and Triple-Barrier).

The Blueprint forbids fixed numeric thresholds. CUSUM's breach threshold and
Triple-Barrier's profit-take / stop-loss multipliers are scaled by this rolling
exponentially-weighted standard deviation so they self-adapt to the local
volatility regime.

**Strictly causal (AFML audit Vulnerability 4):** at index ``t`` the result
depends ONLY on returns at indices ``< t``. We compute the EWM std at ``t``
from returns ``[0..t]`` and then ``.shift(1)`` the output one bar forward, so
``vol[t]`` is determined entirely by information available at ``t-1`` —
``return[t]`` itself plays no role in setting the barriers around an entry at
``t``. Proven by ``tests/unit/phase2/test_volatility.py::
test_volatility_does_not_use_return_at_t``: perturbing ``returns[t]`` leaves
``vol[t]`` numerically unchanged.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def ewm_volatility(
    returns: npt.NDArray[np.float64],
    span: int = 100,
    *,
    min_periods: int | None = None,
    causal_shift: bool = True,
) -> npt.NDArray[np.float64]:
    """Exponentially-weighted standard deviation of a returns series.

    Parameters
    ----------
    returns : 1-D array of returns (NaN allowed; treated as missing by pandas).
    span : EWM span (``α = 2 / (span + 1)``).
    min_periods : minimum non-NaN observations before EWM emits a non-NaN value.
        Defaults to ``span``.
    causal_shift : if True (default), shift the output by one bar so the
        volatility at index ``t`` uses returns strictly before ``t``. This is
        what prevents look-ahead bias in CUSUM and Triple-Barrier thresholds.

    Returns
    -------
    1-D float64 array of the same length as ``returns``.
    """
    if returns.ndim != 1:
        raise ValueError(f"returns must be 1-D; got shape {returns.shape}")
    if min_periods is None:
        min_periods = span
    series = pd.Series(returns)
    vol = series.ewm(span=span, adjust=False, min_periods=min_periods).std()
    if causal_shift:
        vol = vol.shift(1)
    return vol.to_numpy(dtype=np.float64)
