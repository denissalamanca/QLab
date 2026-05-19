"""Phase 8 — StructuralBreakMonitor + event emission (Blueprint §10.3 DoD)."""

from __future__ import annotations

import numpy as np
import pytest

from afml.core.events import Channel, ConceptDriftAlert, MarketRegimeBreak
from afml.monitoring import StructuralBreakMonitor


def _exponential_bubble(n: int, seed: int, *, onset: int, rate: float = 1.04) -> np.ndarray:
    rng = np.random.default_rng(seed)
    series = np.cumsum(rng.standard_normal(n)) + 100.0
    series[onset:] = series[onset] * (rate ** np.arange(n - onset))
    return series


def _mean_reverting(n: int, seed: int, *, phi: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = 0.0
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y + 100.0


@pytest.mark.phase8
def test_monitor_emits_market_regime_break_on_bubble() -> None:
    """Blueprint §10.3 — bubble data ⇒ a populated MarketRegimeBreak event."""
    monitor = StructuralBreakMonitor(n_simulations=99)
    series = _exponential_bubble(120, seed=0, onset=80)
    check = monitor.check_regime("BTCUSD", series, random_state=1)
    assert check.regime_break is True
    assert isinstance(check.event, MarketRegimeBreak)
    assert check.event.channel == Channel.MARKET_REGIME_BREAK
    assert check.event.asset == "BTCUSD"
    assert check.event.test == "gsadf"
    assert check.event.statistic > check.event.critical_value
    assert check.event.producer == "agent_8"


@pytest.mark.phase8
def test_monitor_no_event_on_mean_reverting() -> None:
    monitor = StructuralBreakMonitor(n_simulations=199)
    series = _mean_reverting(150, seed=0)
    check = monitor.check_regime("EURUSD", series, random_state=2)
    assert check.regime_break is False
    assert check.event is None


@pytest.mark.phase8
def test_monitor_emits_concept_drift_alert() -> None:
    """Blueprint §10.2/§10.3 — reversed SHAP importance ⇒ ConceptDriftAlert."""
    monitor = StructuralBreakMonitor()
    train = np.array([0.5, 0.3, 0.1, 0.07, 0.03])
    live = train[::-1].copy()
    check = monitor.check_drift("USA500", train, live)
    assert check.concept_drift is True
    assert isinstance(check.event, ConceptDriftAlert)
    assert check.event.channel == Channel.CONCEPT_DRIFT_ALERT
    assert check.event.asset == "USA500"
    assert check.event.spearman_rank_corr < 0.5


@pytest.mark.phase8
def test_monitor_no_drift_event_when_stable() -> None:
    monitor = StructuralBreakMonitor()
    imp = np.array([0.5, 0.3, 0.1, 0.07, 0.03])
    check = monitor.check_drift("EURUSD", imp, imp.copy())
    assert check.concept_drift is False
    assert check.event is None


@pytest.mark.phase8
def test_regime_event_round_trips_through_pydantic() -> None:
    """The emitted event must serialise + deserialise on the message bus."""
    monitor = StructuralBreakMonitor(n_simulations=99)
    series = _exponential_bubble(120, seed=0, onset=80)
    check = monitor.check_regime("BTCUSD", series, random_state=1)
    assert check.event is not None
    payload = check.event.model_dump_json()
    restored = MarketRegimeBreak.model_validate_json(payload)
    assert restored.asset == "BTCUSD"
    assert restored.test == "gsadf"


@pytest.mark.phase8
def test_monitor_runs_chow_secondary_by_default() -> None:
    """The Chow secondary diagnostic is attached to the regime check."""
    monitor = StructuralBreakMonitor(n_simulations=99, run_chow_secondary=True)
    series = _exponential_bubble(160, seed=0, onset=100)
    check = monitor.check_regime("BTCUSD", series, random_state=1)
    assert check.chow is not None
    # On an explosive series the Chow break should also fire.
    assert check.chow.is_break is True
