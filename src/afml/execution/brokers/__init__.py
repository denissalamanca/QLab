"""Broker adapters — abstract contract + concrete implementations.

- :class:`afml.execution.brokers.base.BrokerAdapter` — the contract every
  broker honours (connect, submit, positions, close, flatten, equity).
- :class:`afml.execution.brokers.mock.InMemoryMockBroker` — deterministic
  simulator for Phases 0-7 tests and dry runs.
- :class:`afml.execution.brokers.mt5.MT5Adapter` — MetaTrader 5 over the
  Wine/VM bridge (Phase 7+, live).
"""

from afml.execution.brokers.base import (
    BrokerAdapter,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)
from afml.execution.brokers.mock import InMemoryMockBroker
from afml.execution.brokers.mt5 import MT5Adapter

__all__ = [
    "BrokerAdapter",
    "Fill",
    "InMemoryMockBroker",
    "MT5Adapter",
    "Order",
    "OrderSide",
    "OrderStatus",
    "Position",
]
