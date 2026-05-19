"""Familywise Error Rate controls — Bonferroni + Holm-Bonferroni.

When ``N`` strategies are tested in parallel, even pure noise produces some
"significant" p-values by chance. FWER controls bound the probability of any
false discovery across the family at a chosen ``α``.

- **Bonferroni** is the simplest and most conservative: reject if
  ``p_i < α / N``. Equivalent to scaling each p-value by ``N``.
- **Holm-Bonferroni** is a step-down variant: order p-values ascending,
  reject ``p_(i)`` iff ``p_(j) < α / (N - j + 1)`` for all ``j ≤ i``. Same
  FWER bound, strictly more powerful than plain Bonferroni.

Both are exposed here as primitives; the Phase 6 orchestrator wires them
into the strategy-filtering step that precedes the DSR computation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

DEFAULT_ALPHA: float = 0.05


def bonferroni_threshold(alpha: float, n_trials: int) -> float:
    """Return the per-trial significance cutoff ``α / N``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be ≥ 1, got {n_trials}")
    return alpha / float(n_trials)


def holm_bonferroni(
    pvalues: npt.NDArray[np.floating],
    alpha: float = DEFAULT_ALPHA,
) -> npt.NDArray[np.bool_]:
    """Holm-Bonferroni step-down rejection mask.

    Parameters
    ----------
    pvalues
        ``(n_trials,)`` array of p-values.
    alpha
        Familywise significance level.

    Returns
    -------
    Boolean mask of shape ``(n_trials,)``: ``True`` ⇒ reject the null
    (the trial is FWER-significant).
    """
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"pvalues must be 1-D, got {p.ndim}-D")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(p)
    sorted_p = p[order]
    # Holm: reject p_(i) if p_(i) < α / (n - i) for the first i where this holds,
    # and reject all earlier.
    thresholds = alpha / (n - np.arange(n, dtype=np.float64))
    reject_sorted = sorted_p < thresholds
    # Step-down: as soon as one fails, everything after fails too.
    first_fail = np.argmax(~reject_sorted) if (~reject_sorted).any() else n
    reject_sorted_final = np.zeros(n, dtype=bool)
    reject_sorted_final[:first_fail] = True

    mask = np.zeros(n, dtype=bool)
    mask[order] = reject_sorted_final
    return mask
