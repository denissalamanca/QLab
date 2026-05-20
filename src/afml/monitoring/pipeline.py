"""Phase 8 orchestrator — the structural-break / drift monitor (Agent 8).

The :class:`StructuralBreakMonitor` is Agent 8's decision core. It runs the
GSADF bubble test and the SHAP drift check, and — when either fires — returns
the corresponding Phase 0 event object ready to publish on the message bus:

- :class:`afml.core.events.MarketRegimeBreak` (channel
  ``afml.alert.regime_break``) when GSADF (or the Chow secondary) rejects.
  Publishing this halts Agent 7 execution.
- :class:`afml.core.events.ConceptDriftAlert` (channel
  ``afml.alert.concept_drift``) when SHAP rank correlation drops below the
  drift threshold.

The monitor is deliberately transport-free: it *produces* the event object,
and the agent runtime publishes it. This keeps the detection logic unit-
testable without a live Redis broker.

**AFML 0-9 polishing audit V2 — GSADF "tick-death" CPU bottleneck.** GSADF is
``O(T²)`` nested OLS regressions; running it synchronously on every raw
``NEW_TICK`` would peg a CPU core, overflow the broker queue, and inject huge
latency into Agent 7. Two rules follow:

1. Agent 8's listener binds to the ``BAR_GENERATED`` (information-bar) event
   from Phase 1 — **never** raw ticks — so the heavy test runs at *bar*
   cadence, not *tick* cadence.
2. Use :meth:`StructuralBreakMonitor.check_regime_async`, which offloads the
   GSADF computation to a worker thread (``loop.run_in_executor``) so the main
   message-broker event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from afml.core.events import ConceptDriftAlert, MarketRegimeBreak
from afml.monitoring.chow import ChowBreakResult, chow_break_test
from afml.monitoring.gsadf import (
    DEFAULT_MIN_WINDOW_FRAC,
    DEFAULT_N_SIMULATIONS,
    BubbleDetectionResult,
    detect_bubble,
)
from afml.monitoring.shap_drift import (
    DEFAULT_DRIFT_THRESHOLD,
    ConceptDriftResult,
    detect_concept_drift,
)

DEFAULT_PRODUCER: str = "agent_8"


@dataclass(frozen=True, slots=True)
class RegimeCheck:
    """Bundled GSADF (+ optional Chow) regime diagnostics for one asset."""

    asset: str
    gsadf: BubbleDetectionResult
    chow: ChowBreakResult | None
    event: MarketRegimeBreak | None

    @property
    def regime_break(self) -> bool:
        return self.event is not None


@dataclass(frozen=True, slots=True)
class DriftCheck:
    """Bundled SHAP-drift diagnostics for one asset."""

    asset: str
    drift: ConceptDriftResult
    event: ConceptDriftAlert | None

    @property
    def concept_drift(self) -> bool:
        return self.event is not None


@dataclass(frozen=True, slots=True)
class StructuralBreakMonitor:
    """Agent 8 monitor — produces regime-break / drift events.

    Parameters
    ----------
    producer
        Event ``producer`` tag (default ``"agent_8"``).
    min_window_frac, n_simulations
        GSADF knobs forwarded to :func:`detect_bubble`.
    drift_threshold
        SHAP rank-correlation threshold (default 0.5).
    run_chow_secondary
        When True, also run the Chow test and attach it to the
        :class:`RegimeCheck` (it does not change the event decision — GSADF
        is primary — but provides a confirming diagnostic).
    """

    producer: str = DEFAULT_PRODUCER
    min_window_frac: float = DEFAULT_MIN_WINDOW_FRAC
    n_simulations: int = DEFAULT_N_SIMULATIONS
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD
    run_chow_secondary: bool = True

    def check_regime(
        self,
        asset: str,
        prices: npt.NDArray[np.floating],
        *,
        random_state: int = 0,
    ) -> RegimeCheck:
        """Run GSADF (primary) + Chow (secondary) on a price series.

        Returns a :class:`RegimeCheck`; ``.event`` is a populated
        :class:`MarketRegimeBreak` iff GSADF rejects the random-walk null.
        """
        gsadf = detect_bubble(
            prices,
            min_window_frac=self.min_window_frac,
            n_simulations=self.n_simulations,
            random_state=random_state,
        )
        chow: ChowBreakResult | None = None
        if self.run_chow_secondary:
            try:
                chow = chow_break_test(prices)
            except ValueError:
                chow = None

        event: MarketRegimeBreak | None = None
        if gsadf.is_bubble:
            event = MarketRegimeBreak(
                producer=self.producer,
                asset=asset,
                test="gsadf",
                statistic=gsadf.gsadf_statistic,
                critical_value=gsadf.critical_value,
            )
        return RegimeCheck(asset=asset, gsadf=gsadf, chow=chow, event=event)

    async def check_regime_async(
        self,
        asset: str,
        prices: npt.NDArray[np.floating],
        *,
        random_state: int = 0,
    ) -> RegimeCheck:
        """Non-blocking :meth:`check_regime` for Agent 8's async listener (V2).

        Offloads the CPU-brutal GSADF sweep to the default thread-pool executor
        via ``loop.run_in_executor`` so the message-broker event loop stays
        responsive (and Agent 7's execution latency is unaffected) while the
        test runs. Returns the identical :class:`RegimeCheck` the synchronous
        path produces.

        Invoke this **only** on ``BAR_GENERATED`` (information-bar) events — not
        per raw tick — so GSADF runs at bar cadence (see the module docstring).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(self.check_regime, asset, prices, random_state=random_state),
        )

    def check_drift(
        self,
        asset: str,
        training_importance: npt.NDArray[np.floating],
        live_importance: npt.NDArray[np.floating],
    ) -> DriftCheck:
        """Run the SHAP rank-correlation drift check.

        Returns a :class:`DriftCheck`; ``.event`` is a populated
        :class:`ConceptDriftAlert` iff the rank correlation falls below the
        drift threshold.
        """
        drift = detect_concept_drift(
            training_importance,
            live_importance,
            threshold=self.drift_threshold,
        )
        event: ConceptDriftAlert | None = None
        if drift.drifted:
            event = ConceptDriftAlert(
                producer=self.producer,
                asset=asset,
                spearman_rank_corr=drift.spearman_rank_corr,
            )
        return DriftCheck(asset=asset, drift=drift, event=event)
