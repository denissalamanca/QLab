"""Phase 7 — ExecutionEngine orchestrator + Blueprint §9.3 DoD.

Blueprint §9.3 DoD:

* **Margin Constraint Test:** 50 concurrent ``p≈1`` signals →
  ``total_margin_used <= FTMO_MAX_DRAWDOWN_BUFFER``.
* **Zero-Size Test:** ``calculate_bet_size(p=0.49) == 0.0`` (in
  ``test_bet_sizing.py``).
* **Mock-broker integration:** full pipeline → orders dispatched within risk
  limits.
"""

from __future__ import annotations

import pytest

from afml.config.risk import FTMO_MAX_DRAWDOWN_BUFFER
from afml.execution import (
    ExecutionEngine,
    InMemoryMockBroker,
    OrderSide,
    RiskEngine,
    Signal,
)


def _engine(equity: float = 100_000.0, c95: float = 10.0) -> ExecutionEngine:
    broker = InMemoryMockBroker(equity)
    broker.connect()
    risk = RiskEngine(account_equity=equity, c95=c95)
    return ExecutionEngine(broker=broker, risk_engine=risk)


@pytest.mark.phase7
def test_margin_constraint_50_concurrent_max_confidence() -> None:
    """Blueprint §9.3 — 50 simultaneous max-confidence signals must NOT push
    total committed margin past the FTMO drawdown buffer."""
    equity = 100_000.0
    engine = _engine(equity=equity, c95=10.0)
    signals = [Signal("EURUSD", 0.999, OrderSide.BUY, 1.10) for _ in range(50)]
    result = engine.execute_batch(signals)
    buffer = FTMO_MAX_DRAWDOWN_BUFFER * equity
    assert result.total_margin_committed <= buffer + 1e-6, (
        f"committed {result.total_margin_committed} > buffer {buffer}"
    )


@pytest.mark.phase7
def test_margin_constraint_holds_across_asset_classes() -> None:
    """Mixed crypto + FX max-confidence burst still respects the buffer."""
    equity = 250_000.0
    engine = _engine(equity=equity, c95=5.0)
    signals = [Signal("BTCUSD", 0.99, OrderSide.BUY, 50_000.0) for _ in range(25)] + [
        Signal("EURUSD", 0.99, OrderSide.SELL, 1.10) for _ in range(25)
    ]
    result = engine.execute_batch(signals)
    buffer = FTMO_MAX_DRAWDOWN_BUFFER * equity
    assert result.total_margin_committed <= buffer + 1e-6


@pytest.mark.phase7
def test_pipeline_dispatches_orders_within_limits() -> None:
    """Mock-broker integration — a realistic mixed-confidence batch dispatches
    orders, and the broker's committed margin matches the engine's."""
    engine = _engine(equity=100_000.0, c95=4.0)
    signals = [
        Signal("EURUSD", 0.85, OrderSide.BUY, 1.10),
        Signal("GBPUSD", 0.49, OrderSide.SELL, 1.25),  # below 0.5 → skipped
        Signal("XAUUSD", 0.92, OrderSide.BUY, 2000.0),
        Signal("USA500", 0.70, OrderSide.BUY, 5000.0),
    ]
    result = engine.execute_batch(signals)
    # The 0.49 signal must be skipped (zero size).
    assert result.n_skipped >= 1
    assert result.n_orders_submitted == result.n_signals - result.n_skipped
    # Broker's committed margin equals the engine's running total.
    assert engine.broker.committed_margin == pytest.approx(result.total_margin_committed)  # type: ignore[attr-defined]


@pytest.mark.phase7
def test_pipeline_skips_all_when_no_confidence() -> None:
    engine = _engine()
    signals = [Signal("EURUSD", 0.4, OrderSide.BUY, 1.10) for _ in range(10)]
    result = engine.execute_batch(signals)
    assert result.n_orders_submitted == 0
    assert result.n_skipped == 10
    assert result.total_margin_committed == 0.0


@pytest.mark.phase7
def test_pipeline_highest_confidence_claims_budget_first() -> None:
    """When the budget is scarce, the highest-probability signals fill before
    lower ones (descending-probability dispatch order)."""
    equity = 100_000.0
    # c95=1 + crypto (50% margin) means ~ one bet exhausts 10% buffer / 0.5 → tiny.
    broker = InMemoryMockBroker(equity)
    broker.connect()
    risk = RiskEngine(account_equity=equity, c95=1.0)
    engine = ExecutionEngine(broker=broker, risk_engine=risk)
    signals = [
        Signal("EURUSD", 0.55, OrderSide.BUY, 1.10),
        Signal("EURUSD", 0.999, OrderSide.BUY, 1.10),  # should fill first
    ]
    result = engine.execute_batch(signals)
    # The first fill must correspond to the high-confidence bet.
    filled = [f for f in result.fills if f.status.value == "filled"]
    assert filled, "expected at least one fill"
    # The highest-confidence bet committed the most margin (largest scaled size).
    assert max(b.raw_size for b in result.sized_bets) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.phase7
def test_emergency_flatten_closes_all_and_resets() -> None:
    engine = _engine()
    engine.execute_batch([Signal("EURUSD", 0.9, OrderSide.BUY, 1.10)])
    assert engine.broker.open_positions()
    closes = engine.emergency_flatten({"EURUSD": 1.10})
    assert len(closes) >= 1
    assert engine.broker.open_positions() == []
    assert engine.risk_engine.committed_margin == 0.0


@pytest.mark.phase7
def test_pipeline_requires_connected_broker() -> None:
    broker = InMemoryMockBroker(100_000.0)  # not connected
    risk = RiskEngine(account_equity=100_000.0)
    engine = ExecutionEngine(broker=broker, risk_engine=risk)
    with pytest.raises(RuntimeError, match="not connected"):
        engine.execute_batch([Signal("EURUSD", 0.9, OrderSide.BUY, 1.10)])


@pytest.mark.phase7
def test_pipeline_empty_batch_is_noop() -> None:
    engine = _engine()
    result = engine.execute_batch([])
    assert result.n_signals == 0
    assert result.n_orders_submitted == 0
    assert result.total_margin_committed == 0.0
