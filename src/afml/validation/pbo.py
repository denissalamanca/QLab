"""Probability of Backtest Overfitting (Bailey & López de Prado 2014).

The PBO statistic estimates the probability that a strategy chosen as the
in-sample best-performer turns out to be below-median out-of-sample —
i.e., the chance our "winning" strategy was a fluke. AFML treats
``PBO < 5%`` as the hard ship gate.

Algorithm (Bailey-Lopez de Prado §6 / López de Prado 2018 Ch. 14):

1. For each combinatorial split ``c`` of the CPCV (or CSCV) procedure, hold
   the in-sample performance vector ``IS[c, s]`` and out-of-sample
   performance vector ``OOS[c, s]`` for ``s`` in the candidate strategy
   set.
2. Identify ``n*(c) = argmax_s IS[c, s]`` — the in-sample winner for split
   ``c``.
3. Compute the OOS rank of ``n*(c)`` among all strategies:
   ``r*(c) = rank(OOS[c, n*(c)]) / N``.
4. Map to logit: ``λ(c) = log(ω) where ω = r*(c) / (1 - r*(c))``.
   Boundary handling: ranks are mapped to ``(0, 1)`` exclusive via
   continuity correction ``r → (r - 0.5) / N`` then clipped to ``ε``.
5. ``PBO = P(λ ≤ 0) = #{c : λ(c) ≤ 0} / #splits``.

A strategy with PBO ≈ 0.5 is no better than chance; PBO ≪ 0.5 indicates
genuine OOS edge; PBO ≫ 0.5 indicates *anti-skill* (IS-best maps to
OOS-worst — a classic overfitting fingerprint).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

DEFAULT_RANK_EPSILON: float = 1e-6


@dataclass(frozen=True, slots=True)
class PBOResult:
    """Diagnostic record from :func:`compute_pbo`.

    Attributes
    ----------
    pbo
        Probability of Backtest Overfitting — the fraction of CPCV splits
        where the IS-winning strategy ranked below median OOS.
    n_splits
        Number of combinatorial splits in the underlying matrix.
    n_strategies
        Width of the candidate strategy set.
    logits
        Per-split logit values ``λ(c)`` — the raw input to the PBO mean.
    """

    pbo: float
    n_splits: int
    n_strategies: int
    logits: npt.NDArray[np.float64]


def compute_pbo(
    in_sample_perf: npt.NDArray[np.floating],
    out_sample_perf: npt.NDArray[np.floating],
    *,
    rank_epsilon: float = DEFAULT_RANK_EPSILON,
) -> PBOResult:
    """Bailey & López de Prado 2014 PBO from CPCV in-sample / out-of-sample matrices.

    Parameters
    ----------
    in_sample_perf, out_sample_perf
        Both ``(n_splits, n_strategies)`` arrays. Each row corresponds to one
        CPCV combinatorial split; each column to one candidate strategy.
        Identical row dimensions required.
    rank_epsilon
        Lower / upper clip on the continuity-corrected rank to avoid
        ``log(0)`` and ``log(∞)`` at the extremes.

    Returns
    -------
    :class:`PBOResult`.

    Notes
    -----
    The continuity-corrected rank formula ``(rank - 0.5) / N`` maps integer
    ranks ``1..N`` to ``(0.5/N, 1 - 0.5/N)`` — strictly inside ``(0, 1)`` so
    the subsequent ``log(ω)`` is finite. Ties are broken via average rank
    (``scipy.stats.rankdata``'s default).
    """
    is_arr = np.asarray(in_sample_perf, dtype=np.float64)
    oos_arr = np.asarray(out_sample_perf, dtype=np.float64)
    if is_arr.shape != oos_arr.shape:
        raise ValueError(f"shape mismatch: in_sample={is_arr.shape}, out_sample={oos_arr.shape}")
    if is_arr.ndim != 2:
        raise ValueError(f"performance arrays must be 2-D, got {is_arr.ndim}-D")
    n_splits, n_strategies = is_arr.shape
    if n_strategies < 2:
        raise ValueError(f"need ≥ 2 candidate strategies for PBO, got {n_strategies}")
    if n_splits < 1:
        raise ValueError("need ≥ 1 CPCV split")
    if not np.all(np.isfinite(is_arr)) or not np.all(np.isfinite(oos_arr)):
        raise ValueError("performance arrays must be finite")

    logits = np.empty(n_splits, dtype=np.float64)
    for c in range(n_splits):
        n_star = int(np.argmax(is_arr[c]))
        # Continuity-corrected OOS rank of n_star.
        oos_ranks = rankdata(oos_arr[c], method="average")
        r_star = (oos_ranks[n_star] - 0.5) / n_strategies
        r_star = float(np.clip(r_star, rank_epsilon, 1.0 - rank_epsilon))
        omega = r_star / (1.0 - r_star)
        logits[c] = np.log(omega)

    pbo = float(np.mean(logits <= 0.0))
    return PBOResult(
        pbo=pbo,
        n_splits=n_splits,
        n_strategies=n_strategies,
        logits=logits,
    )
