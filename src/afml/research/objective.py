"""Plateau objective ``s(g)`` — median PWF OOS strategy Sharpe (Ops M1.5).

The approved surface objective (CEO decision #5): for each PurgedWalkForward
out-of-sample fold, turn Brain-2's calibrated ``P(success)`` into a bet size,
realise the strategy return per event (``bet · side · realised return``),
annualise the fold's Sharpe, and take the **median across folds**. Median (not
mean) is robust to one lucky fold.

Consumes the opt-in ``BrainTwoResult.oos_predictions`` (per-fold holdout indices
+ calibrated probabilities) added to Phase 5 for exactly this purpose — no
re-fitting.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from afml.execution import calculate_bet_size
from afml.modeling import BrainTwoResult

#: A fold's strategy-return standard deviation below this is treated as zero
#: (degenerate fold → no finite Sharpe).
_MIN_STD: float = 1e-12


def oos_strategy_sharpe(
    result: BrainTwoResult,
    return_pct: npt.NDArray[np.float64],
    side_sign: npt.NDArray[np.int64],
    *,
    periods_per_year: float,
    min_fold_events: int = 2,
) -> float | None:
    """Median across PWF OOS folds of the bet-sized strategy's annualised Sharpe.

    Parameters
    ----------
    result
        A :class:`BrainTwoResult` from ``train_brain_two(..., collect_oos_predictions=True)``.
    return_pct
        ``(n_events,)`` realised barrier-touch return per event (``TripleBarrierLabels``
        ``return_pct``), aligned row-for-row with the Brain-2 ``X``.
    side_sign
        ``(n_events,)`` primary-signal direction as ``+1`` (long) / ``-1`` (short),
        aligned with ``return_pct``.
    periods_per_year
        Annualisation factor — the asset/config's realised events-per-year
        (reused unchanged by the M3 OOS gate).
    min_fold_events
        Folds with fewer holdout events than this are skipped.

    Returns
    -------
    The median fold Sharpe, or ``None`` if no fold yields a finite Sharpe
    (e.g. a halted-MDA result, or zero-variance returns everywhere).
    """
    fold_sharpes = oos_strategy_sharpe_per_fold(
        result,
        return_pct,
        side_sign,
        periods_per_year=periods_per_year,
        min_fold_events=min_fold_events,
    )
    if not fold_sharpes:
        return None
    return float(np.median(fold_sharpes))


def oos_strategy_sharpe_per_fold(
    result: BrainTwoResult,
    return_pct: npt.NDArray[np.float64],
    side_sign: npt.NDArray[np.int64],
    *,
    periods_per_year: float,
    min_fold_events: int = 2,
) -> list[float]:
    """Per-fold annualised strategy Sharpe (the list :func:`oos_strategy_sharpe` medians).

    Exposed for diagnostics: the *dispersion* across folds is itself a signal —
    a high median riding on one lucky fold (wide spread) is far less trustworthy
    than a tight cluster. Same semantics/guards as :func:`oos_strategy_sharpe`.
    """
    if result.oos_predictions is None:
        raise ValueError(
            "BrainTwoResult carries no oos_predictions — call "
            "train_brain_two(..., collect_oos_predictions=True)"
        )
    if periods_per_year <= 0.0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")

    annualisation = float(np.sqrt(periods_per_year))
    fold_sharpes: list[float] = []
    for fold in result.oos_predictions:
        idx = fold.holdout_indices
        if idx.size < min_fold_events:
            continue
        bets = np.array(
            [calculate_bet_size(float(p)) for p in fold.calibrated_proba], dtype=np.float64
        )
        strat_returns = bets * side_sign[idx].astype(np.float64) * return_pct[idx]
        std = float(strat_returns.std(ddof=1)) if strat_returns.size > 1 else 0.0
        if not np.isfinite(std) or std < _MIN_STD:
            continue
        fold_sharpes.append(float(strat_returns.mean() / std) * annualisation)

    return fold_sharpes
