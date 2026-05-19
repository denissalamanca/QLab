"""Orthogonality check — reject signal vectors that duplicate a deployed strategy.

Blueprint §4.3 (Orthogonality Test): "Pearson correlation of new signal vector
with all deployed vectors in Alpha Registry must be ≤ threshold."

The signal vector here is the binary event-presence indicator on a canonical
bar grid (1 if the family fired at bar t, 0 otherwise) OR a side-signed
indicator (+1 / -1 / 0). Both work; the test cares only about correlation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def max_correlation(
    new_signal: npt.NDArray[np.float64],
    existing_signals: Sequence[npt.NDArray[np.float64]],
) -> float:
    """Maximum absolute Pearson correlation between ``new_signal`` and each
    member of ``existing_signals``.

    If the arrays differ in length, both are truncated to the shorter length
    (the prefix is shared on a canonical bar grid). Zero-variance signals are
    skipped (correlation undefined). Returns ``0.0`` when ``existing_signals``
    is empty.
    """
    if len(existing_signals) == 0:
        return 0.0

    correlations: list[float] = []
    for existing in existing_signals:
        n = min(new_signal.shape[0], existing.shape[0])
        if n < 2:
            continue
        a = new_signal[:n].astype(np.float64, copy=False)
        b = existing[:n].astype(np.float64, copy=False)
        if np.std(a) == 0.0 or np.std(b) == 0.0:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if np.isnan(c):
            continue
        correlations.append(abs(c))

    return max(correlations) if correlations else 0.0


def is_orthogonal(
    new_signal: npt.NDArray[np.float64],
    existing_signals: Sequence[npt.NDArray[np.float64]],
    *,
    threshold: float = 0.5,
) -> bool:
    """``True`` iff ``max_correlation`` is strictly less than ``threshold``.

    The default ``threshold`` of 0.5 means: any new signal that shares more than
    half of its variance with an existing deployed strategy is rejected.
    Configurable per-asset / per-family if needed; the Blueprint leaves the
    exact threshold as an operational decision.
    """
    return max_correlation(new_signal, existing_signals) < threshold
