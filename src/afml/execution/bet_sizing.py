"""Probabilistic bet sizing (Blueprint §9.1, López de Prado 2018 Ch. 10).

Brain 2 emits a calibrated probability ``p = P(success | features)``. We turn
that into a bet size in ``[0, 1]`` via the AFML test-statistic transform:

::

    z    = (p - 0.5) / sqrt(p · (1 - p))      # ~ N(0, 1) under H0: p = 0.5
    size = 2 · Φ(z) - 1                        # ∈ (0, 1) for p > 0.5

with the meta-labelling convention that ``p ≤ 0.5 ⇒ size = 0`` (the meta-model
is not confident the primary signal will succeed, so we stand aside).

The size grows monotonically from 0 (at ``p = 0.5``) toward 1 (as ``p → 1``):
``p = 0.6 → 0.16``, ``p = 0.75 → 0.50``, ``p = 0.9 → 0.82``.

**Mixture-of-Gaussians fallback (Blueprint §9.1 / implementation plan).** The
CDF transform assumes the per-bet ``z`` statistic is approximately Gaussian.
When a *batch* of probabilities produces a ``z``-distribution that fails a
Shapiro-Wilk normality test (``p_value < 0.05``), the Gaussian CDF
mis-sizes the tails. In that regime we fit a 2-component Gaussian mixture to
the observed ``z`` values and size each bet by the mixture CDF instead. This
keeps the sizing calibrated when Brain 2's probabilities are bimodal (a common
signature of regime mixing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm, shapiro
from sklearn.mixture import GaussianMixture

# Below this probability we do not trade (meta-label says "skip").
NO_TRADE_THRESHOLD: float = 0.5
# Clip probabilities away from {0, 1} so the z-score denominator stays finite.
PROBA_EPSILON: float = 1e-6
# Shapiro-Wilk significance — below this the z-batch is "not Gaussian".
SHAPIRO_ALPHA: float = 0.05
# Minimum batch size for a meaningful Shapiro-Wilk test.
MIN_SHAPIRO_SAMPLES: int = 20
MOG_N_COMPONENTS: int = 2


def bet_size_from_probability(p: float) -> float:
    """Single-probability bet size via the Gaussian-CDF transform.

    Parameters
    ----------
    p
        Calibrated success probability from Brain 2, in ``[0, 1]``.

    Returns
    -------
    Bet size in ``[0, 1]``. Exactly ``0.0`` when ``p ≤ 0.5`` (Blueprint §9.1
    zero-size rule, verified by ``calculate_bet_size(p=0.49) == 0.0``).

    Raises
    ------
    ValueError
        If ``p`` is outside ``[0, 1]`` or non-finite.
    """
    if not np.isfinite(p):
        raise ValueError(f"probability must be finite, got {p}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p}")
    if p <= NO_TRADE_THRESHOLD:
        return 0.0
    p_clipped = min(p, 1.0 - PROBA_EPSILON)
    z = (p_clipped - 0.5) / np.sqrt(p_clipped * (1.0 - p_clipped))
    size = 2.0 * float(norm.cdf(z)) - 1.0
    return max(0.0, min(1.0, size))


# Public DoD alias — the Blueprint names the entry point ``calculate_bet_size``.
def calculate_bet_size(p: float) -> float:
    """Alias for :func:`bet_size_from_probability` (Blueprint §9.3 naming)."""
    return bet_size_from_probability(p)


def _z_scores(probabilities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Vectorised AFML z-statistic, with the ``p ≤ 0.5`` rows zeroed."""
    p = np.clip(probabilities, PROBA_EPSILON, 1.0 - PROBA_EPSILON)
    z = (p - 0.5) / np.sqrt(p * (1.0 - p))
    return np.asarray(z, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BatchBetSizes:
    """Output of :func:`bet_sizes_for_batch`.

    Attributes
    ----------
    sizes
        ``(n,)`` bet sizes in ``[0, 1]``; rows with ``p ≤ 0.5`` are ``0``.
    used_mixture_fallback
        Whether the Mixture-of-Gaussians CDF was used instead of the
        standard-normal CDF (triggered by a failed Shapiro-Wilk test).
    shapiro_pvalue
        The Shapiro-Wilk p-value on the active (``p > 0.5``) z-scores, or
        ``nan`` when too few active bets to test.
    """

    sizes: npt.NDArray[np.float64]
    used_mixture_fallback: bool
    shapiro_pvalue: float


def bet_sizes_for_batch(
    probabilities: npt.NDArray[np.floating],
    *,
    shapiro_alpha: float = SHAPIRO_ALPHA,
    random_state: int = 0,
) -> BatchBetSizes:
    """Size a batch of bets, falling back to a Gaussian mixture if non-normal.

    Parameters
    ----------
    probabilities
        ``(n,)`` calibrated success probabilities.
    shapiro_alpha
        Shapiro-Wilk significance threshold. If the active z-scores'
        p-value falls below this, the mixture-CDF fallback engages.
    random_state
        Seeds the ``GaussianMixture`` EM fit for reproducibility.

    Returns
    -------
    :class:`BatchBetSizes`.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError(f"probabilities must be 1-D, got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")

    active_mask = probs > NO_TRADE_THRESHOLD
    sizes = np.zeros_like(probs)
    shapiro_pvalue = float("nan")
    used_mixture = False

    if not active_mask.any():
        return BatchBetSizes(
            sizes=sizes, used_mixture_fallback=False, shapiro_pvalue=shapiro_pvalue
        )

    active_probs = probs[active_mask]
    active_z = _z_scores(active_probs)

    # Decide whether the Gaussian-CDF assumption holds for this batch.
    if active_z.size >= MIN_SHAPIRO_SAMPLES and np.std(active_z) > 0.0:
        shapiro_pvalue = float(shapiro(active_z).pvalue)
        used_mixture = shapiro_pvalue < shapiro_alpha

    if used_mixture:
        gm = GaussianMixture(n_components=MOG_N_COMPONENTS, random_state=random_state)
        gm.fit(active_z.reshape(-1, 1))
        active_sizes = _mixture_cdf_sizes(active_z, gm)
    else:
        active_sizes = 2.0 * norm.cdf(active_z) - 1.0

    sizes[active_mask] = np.clip(active_sizes, 0.0, 1.0)
    return BatchBetSizes(
        sizes=sizes,
        used_mixture_fallback=used_mixture,
        shapiro_pvalue=shapiro_pvalue,
    )


def _mixture_cdf_sizes(
    z: npt.NDArray[np.float64],
    gm: GaussianMixture,
) -> npt.NDArray[np.float64]:
    """Bet sizes from a fitted 2-component Gaussian mixture CDF.

    The mixture CDF is the weight-averaged sum of the component normal CDFs;
    the bet size is ``2 · F_mix(z) - 1`` mirroring the single-Gaussian map.
    """
    weights = gm.weights_.ravel()
    means = gm.means_.ravel()
    stds = np.sqrt(gm.covariances_.ravel())
    # Mixture CDF evaluated at each z.
    mix_cdf = np.zeros_like(z)
    for w, mu, sd in zip(weights, means, stds, strict=True):
        mix_cdf += w * norm.cdf(z, loc=mu, scale=max(sd, PROBA_EPSILON))
    return np.asarray(2.0 * mix_cdf - 1.0, dtype=np.float64)
