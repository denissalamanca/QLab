"""Deflated Sharpe Ratio + Expected Maximum Sharpe under multiple testing.

Bailey & López de Prado 2014, "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality".

The "Deflated" Sharpe is the probability that the observed Sharpe is
*greater than zero* after correcting for:

1. **Multiple testing.** ``E[max{SR}]`` grows with the number of trials
   tested even when none of them are genuinely profitable. The deflation
   subtracts this expected best-case-under-null from the observed SR.
2. **Higher moments.** The standard Sharpe asymptotic assumes IID Gaussian
   returns; financial returns are typically left-skewed and heavy-tailed.
   The DSR's standard-error formula incorporates skewness ``γ_3`` and
   kurtosis ``γ_4`` of the return distribution.

The deflated null-hypothesis test statistic is:

::

    DSR = Φ( (SR - E[max{SR}]) · √(T - 1) /
              √(1 - γ_3·SR + (γ_4 - 1)/4 · SR²) )

where ``Φ`` is the standard normal CDF, ``T`` is the number of observations,
``γ_3`` / ``γ_4`` are sample skewness / kurtosis. AFML treats ``DSR > 1.0``
operationally as ``> 0.95`` — i.e. a 95% probability the Sharpe is genuinely
positive after multiple-testing deflation. We surface both forms so the
caller can pick whichever convention they prefer.

The Expected Max Sharpe (the deflation amount) is the parametric formula:

::

    E[max{SR_n}] ≈ E[SR] + σ_SR · ( Z(1 - 1/n)·(1 - γ) + Z(1 - 1/(n·e))·γ )

where ``Z`` is the inverse-normal CDF and ``γ`` is the Euler-Mascheroni
constant (``≈ 0.5772``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

EULER_MASCHERONI: float = 0.5772156649015329


@dataclass(frozen=True, slots=True)
class DSRResult:
    """Output of :func:`deflated_sharpe_ratio`.

    Attributes
    ----------
    dsr
        Probability ``P(SR > 0)`` after multiple-testing + non-normality
        deflation. Range ``[0, 1]``.
    sharpe_observed
        The strategy's raw Sharpe ratio on the OOS path(s).
    expected_max_sharpe
        Deflation amount — the parametric estimate of the maximum Sharpe
        across ``n_trials`` independent Gaussian SR draws.
    n_trials
        Multiple-testing denominator (from the Alpha Registry trial count).
    n_observations
        Length of the return series feeding the Sharpe.
    """

    dsr: float
    sharpe_observed: float
    expected_max_sharpe: float
    n_trials: int
    n_observations: int


def expected_max_sharpe(
    n_trials: int,
    sharpe_std: float,
    *,
    sharpe_mean: float = 0.0,
) -> float:
    """Parametric ``E[max{SR_n}]`` (Bailey & López de Prado 2014 eqn. 4).

    Parameters
    ----------
    n_trials
        Number of independent strategies tested. Must be ≥ 2 (for ``n = 1``
        the maximum trivially equals the single trial's SR — degenerate).
    sharpe_std
        Sample standard deviation of the trial SRs (or a prior, when the
        population isn't observed).
    sharpe_mean
        Optional prior on ``E[SR]`` across trials. Defaults to 0 (the null
        hypothesis under multiple testing).

    Returns
    -------
    The expected maximum SR. Multiply ``sharpe_std`` to scale.
    """
    if n_trials < 2:
        raise ValueError(f"n_trials must be ≥ 2, got {n_trials}")
    if sharpe_std < 0:
        raise ValueError(f"sharpe_std must be ≥ 0, got {sharpe_std}")
    n_f = float(n_trials)
    z_a = norm.ppf(1.0 - 1.0 / n_f)
    z_b = norm.ppf(1.0 - 1.0 / (n_f * np.e))
    return float(
        sharpe_mean + sharpe_std * (z_a * (1.0 - EULER_MASCHERONI) + z_b * EULER_MASCHERONI)
    )


def deflated_sharpe_ratio(
    returns: npt.NDArray[np.floating],
    n_trials: int,
    *,
    sharpe_std_of_trials: float | None = None,
    periods_per_year: int = 252,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio from a return series.

    Parameters
    ----------
    returns
        ``(n_observations,)`` per-period strategy returns. Need not be
        annualized; the function annualizes internally based on
        ``periods_per_year``.
    n_trials
        Number of strategies considered when picking this one. The Alpha
        Registry's ``total_trials()`` is the canonical source (every
        hypothesis, including ``FAILED_AT_MDA`` ones, counts).
    sharpe_std_of_trials
        Cross-sectional standard deviation of the trial Sharpe ratios.
        If ``None``, defaults to ``1.0`` (the Bailey-Lopez de Prado
        "uninformative" prior — yields the most conservative deflation).
    periods_per_year
        Annualisation factor (252 = daily trading days; 12 = monthly;
        1 = pre-annualised).

    Returns
    -------
    :class:`DSRResult`.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1 or r.size < 2:
        raise ValueError(f"returns must be 1-D with ≥ 2 obs, got shape {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns must be finite")
    if n_trials < 1:
        raise ValueError(f"n_trials must be ≥ 1, got {n_trials}")

    n = r.size
    mean = float(r.mean())
    std = float(r.std(ddof=1))
    # numpy 2-pass std leaves ~ULP-level residual when the series is constant;
    # treat anything below 1e-12 as zero-variance / degenerate.
    if std < 1e-12:
        # Degenerate — assign DSR = 0.5 (no signal either way).
        return DSRResult(
            dsr=0.5,
            sharpe_observed=0.0,
            expected_max_sharpe=0.0,
            n_trials=n_trials,
            n_observations=n,
        )

    # Per-period Sharpe (no annualization yet — Bailey-Lopez de Prado works
    # on the period-frequency SR; we'll report the annualized SR for
    # readability but the deflation math is on the unscaled value).
    sharpe_period = mean / std
    sharpe_annualized = sharpe_period * np.sqrt(periods_per_year)

    # Higher moments.
    z = (r - mean) / std
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))  # raw 4th moment; kurtosis with offset 3 not used here.

    # E[max{SR}] deflation amount, in per-period SR units.
    sharpe_std_input = sharpe_std_of_trials if sharpe_std_of_trials is not None else 1.0
    e_max_sr = expected_max_sharpe(max(n_trials, 2), sharpe_std_input) if n_trials >= 2 else 0.0

    # Variance of the SR estimator (per AFML eqn., adjusts for higher moments).
    denom = 1.0 - skew * sharpe_period + (kurt - 1.0) / 4.0 * sharpe_period**2
    denom = max(denom, 1e-12)
    se = np.sqrt(denom / (n - 1))

    z_stat = (sharpe_period - e_max_sr) / se
    dsr = float(norm.cdf(z_stat))
    return DSRResult(
        dsr=dsr,
        sharpe_observed=float(sharpe_annualized),
        expected_max_sharpe=float(e_max_sr * np.sqrt(periods_per_year)),
        n_trials=n_trials,
        n_observations=n,
    )
