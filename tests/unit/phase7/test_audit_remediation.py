"""Phase 0-8 final audit — Phase 7 patches (V1 zero-div, V4 rehydration)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from afml.execution.bet_sizing import bet_sizes_for_batch, calculate_bet_size
from afml.execution.brokers import InMemoryMockBroker, Order, OrderSide
from afml.execution.pipeline import ExecutionEngine, Signal
from afml.execution.risk import RiskEngine


# ---------------------------------------------------------------- V1: zero-div
@pytest.mark.phase7
def test_bet_size_at_probability_one_does_not_crash() -> None:
    """AFML 0-8 audit V1 — a perfectly-certain Brain 2 (p = 1.0) must NOT
    raise ZeroDivisionError; the size clips to a finite value near 1.0."""
    size = calculate_bet_size(1.0)
    assert np.isfinite(size)
    assert 0.99 < size <= 1.0


@pytest.mark.phase7
def test_bet_size_at_probability_zero_is_zero() -> None:
    """p = 0.0 ≤ 0.5 → no trade (and definitely no division-by-zero)."""
    assert calculate_bet_size(0.0) == 0.0


@pytest.mark.phase7
def test_bet_size_monotone_through_extreme() -> None:
    """Sizes increase monotonically up to p = 1.0 with no discontinuity."""
    probs = [0.51, 0.6, 0.75, 0.9, 0.99, 0.999, 1.0]
    sizes = [calculate_bet_size(p) for p in probs]
    assert all(np.isfinite(s) for s in sizes)
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(sizes))


@pytest.mark.phase7
def test_batch_bet_sizing_handles_probability_one() -> None:
    """A batch containing p = 1.0 (and 0.0) must size without crashing."""
    probs = np.array([0.0, 0.5, 0.51, 0.9, 1.0])
    result = bet_sizes_for_batch(probs)
    assert np.all(np.isfinite(result.sizes))
    assert result.sizes[-1] > 0.99  # p=1.0 → near-max size
    assert result.sizes[0] == 0.0  # p=0.0 → skip


# -------------------------------------------------------------- V4: rehydration
@pytest.mark.phase7
def test_rehydrate_state_restores_committed_margin() -> None:
    """AFML 0-8 audit V4 — on startup the engine queries the broker for open
    positions and seeds the risk engine's committed margin, so it cannot
    over-size having "forgotten" the live book."""
    broker = InMemoryMockBroker(starting_equity=100_000.0)
    broker.connect()
    # Pre-existing open positions (as if from before a restart).
    broker.submit_order(Order("EURUSD", OrderSide.BUY, size=1.0, margin=3000.0), 1.10)
    broker.submit_order(Order("XAUUSD", OrderSide.BUY, size=1.0, margin=2000.0), 1900.0)

    # Fresh engine after "restart" — committed margin starts at 0.
    risk = RiskEngine(account_equity=100_000.0, c95=5.0)
    engine = ExecutionEngine(broker=broker, risk_engine=risk)
    assert risk.committed_margin == 0.0

    rehydration = engine.rehydrate_state()
    assert rehydration.n_open_positions == 2
    assert rehydration.rehydrated_margin == pytest.approx(5000.0)
    # The risk engine now reflects the live book.
    assert risk.committed_margin == pytest.approx(5000.0)
    assert risk.remaining_budget == pytest.approx(risk.margin_budget - 5000.0)


@pytest.mark.phase7
def test_rehydration_prevents_oversizing_after_restart() -> None:
    """Without rehydration the engine would size a new burst against the full
    buffer; WITH rehydration the already-committed margin is respected so
    total committed never breaches the FTMO buffer."""
    equity = 100_000.0
    # Broker already holds positions consuming most of the 10% buffer.
    broker = InMemoryMockBroker(starting_equity=equity)
    broker.connect()
    broker.submit_order(Order("EURUSD", OrderSide.BUY, size=1.0, margin=9000.0), 1.10)

    risk = RiskEngine(account_equity=equity, c95=1.0)  # buffer = 10% = 10_000
    engine = ExecutionEngine(broker=broker, risk_engine=risk)
    engine.rehydrate_state()  # committed ← 9000

    # A new max-confidence signal arrives. Remaining budget is only 1000.
    result = engine.execute_batch([
        Signal("BTCUSD", probability=0.999, side=OrderSide.BUY, reference_price=50_000.0)
    ])
    # Total committed (rehydrated 9000 + new bet) must not exceed the buffer.
    assert risk.committed_margin <= risk.margin_budget + 1e-6
    assert result.total_margin_committed <= risk.margin_budget + 1e-6


@pytest.mark.phase7
def test_get_open_positions_alias_matches_open_positions() -> None:
    broker = InMemoryMockBroker(starting_equity=50_000.0)
    broker.connect()
    broker.submit_order(Order("EURUSD", OrderSide.BUY, size=1.0, margin=1000.0), 1.10)
    assert broker.get_open_positions() == broker.open_positions()
