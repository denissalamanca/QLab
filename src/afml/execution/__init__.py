"""Phase 7 — Probabilistic Bet Sizing & Execution (Blueprint §9).

Public API:

- **Bet sizing** (:mod:`afml.execution.bet_sizing`): ``calculate_bet_size`` /
  ``bet_size_from_probability`` (z-score → Gaussian CDF; 0 when ``p ≤ 0.5``);
  ``bet_sizes_for_batch`` with the Mixture-of-Gaussians fallback for
  non-Gaussian batches.
- **Risk engine** (:mod:`afml.execution.risk`): ``RiskEngine`` — concurrent
  ``c_95`` scaling, ESMA leverage caps, and the FTMO drawdown-buffer hard cap.
- **Brokers** (:mod:`afml.execution.brokers`): ``BrokerAdapter`` contract,
  ``InMemoryMockBroker`` simulator, ``MT5Adapter`` live scaffold.
- **Orchestrator** (:mod:`afml.execution.pipeline`): ``ExecutionEngine`` —
  signals → sized bets → risk → broker dispatch, plus ``emergency_flatten``.
"""

from afml.execution.bet_sizing import (
    BatchBetSizes,
    bet_size_from_probability,
    bet_sizes_for_batch,
    calculate_bet_size,
)
from afml.execution.brokers import (
    BrokerAdapter,
    Fill,
    InMemoryMockBroker,
    MT5Adapter,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)
from afml.execution.pipeline import (
    DispatchResult,
    ExecutionEngine,
    Signal,
)
from afml.execution.risk import RiskEngine, SizedBet

__all__ = [
    "BatchBetSizes",
    "BrokerAdapter",
    "DispatchResult",
    "ExecutionEngine",
    "Fill",
    "InMemoryMockBroker",
    "MT5Adapter",
    "Order",
    "OrderSide",
    "OrderStatus",
    "Position",
    "RiskEngine",
    "Signal",
    "SizedBet",
    "bet_size_from_probability",
    "bet_sizes_for_batch",
    "calculate_bet_size",
]
