"""Phase 7 — broker adapters (mock lifecycle + MT5 scaffold guards)."""

from __future__ import annotations

import pytest

from afml.execution.brokers import (
    InMemoryMockBroker,
    MT5Adapter,
    Order,
    OrderSide,
    OrderStatus,
)


@pytest.mark.phase7
def test_mock_broker_requires_connect() -> None:
    broker = InMemoryMockBroker(100_000.0)
    order = Order("EURUSD", OrderSide.BUY, size=0.1, margin=1000.0)
    with pytest.raises(RuntimeError, match="not connected"):
        broker.submit_order(order, reference_price=1.10)


@pytest.mark.phase7
def test_mock_broker_fills_and_tracks_position() -> None:
    broker = InMemoryMockBroker(100_000.0)
    broker.connect()
    order = Order("EURUSD", OrderSide.BUY, size=0.1, margin=1000.0)
    fill = broker.submit_order(order, reference_price=1.10)
    assert fill.status == OrderStatus.FILLED
    assert fill.fill_price == 1.10
    assert broker.committed_margin == pytest.approx(1000.0)
    positions = broker.open_positions()
    assert len(positions) == 1
    assert positions[0].asset == "EURUSD"


@pytest.mark.phase7
def test_mock_broker_rejects_zero_size() -> None:
    broker = InMemoryMockBroker(100_000.0)
    broker.connect()
    order = Order("EURUSD", OrderSide.BUY, size=0.0, margin=0.0)
    fill = broker.submit_order(order, reference_price=1.10)
    assert fill.status == OrderStatus.REJECTED
    assert broker.open_positions() == []


@pytest.mark.phase7
def test_mock_broker_rejects_over_budget() -> None:
    broker = InMemoryMockBroker(100_000.0, margin_budget=5_000.0)
    broker.connect()
    fill1 = broker.submit_order(Order("EURUSD", OrderSide.BUY, 0.1, 4_000.0), 1.10)
    assert fill1.status == OrderStatus.FILLED
    fill2 = broker.submit_order(Order("GBPUSD", OrderSide.BUY, 0.1, 3_000.0), 1.25)
    assert fill2.status == OrderStatus.REJECTED  # would exceed 5000 budget
    assert broker.committed_margin == pytest.approx(4_000.0)


@pytest.mark.phase7
def test_mock_broker_close_realizes_pnl() -> None:
    broker = InMemoryMockBroker(100_000.0)
    broker.connect()
    broker.submit_order(Order("EURUSD", OrderSide.BUY, 10_000.0, 1_000.0), reference_price=1.10)
    # Price rises 1.10 → 1.11; long PnL = +0.01 * 10000 = +100.
    close = broker.close_position("EURUSD", reference_price=1.11)
    assert close is not None
    assert close.status == OrderStatus.CLOSED
    assert broker.equity() == pytest.approx(100_000.0 + 100.0)
    assert broker.open_positions() == []
    assert broker.committed_margin == pytest.approx(0.0)


@pytest.mark.phase7
def test_mock_broker_flatten_all() -> None:
    broker = InMemoryMockBroker(100_000.0)
    broker.connect()
    broker.submit_order(Order("EURUSD", OrderSide.BUY, 1.0, 1_000.0), 1.10)
    broker.submit_order(Order("XAUUSD", OrderSide.SELL, 1.0, 2_000.0), 2000.0)
    closes = broker.flatten_all({"EURUSD": 1.10, "XAUUSD": 2000.0})
    assert len(closes) == 2
    assert broker.open_positions() == []
    assert broker.committed_margin == pytest.approx(0.0)


@pytest.mark.phase7
def test_mt5_adapter_connect_fails_without_terminal() -> None:
    """In CI / local dev the MetaTrader5 package + terminal are absent, so
    connect() must raise a clear, actionable error rather than import-crash."""
    adapter = MT5Adapter(login=1, password="x", server="demo")
    assert not adapter.is_connected()
    with pytest.raises(RuntimeError, match=r"MetaTrader5|not connected|unavailable"):
        adapter.connect()


@pytest.mark.phase7
def test_mt5_adapter_methods_guard_unconnected() -> None:
    adapter = MT5Adapter(login=1, password="x", server="demo")
    with pytest.raises(RuntimeError, match="not connected"):
        adapter.open_positions()
