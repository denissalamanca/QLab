"""``BrokerAdapter`` contract + order / position / fill value objects.

Every broker — the in-memory mock (Phases 0-7) and the live MetaTrader 5
bridge (Phase 7+) — implements this exact interface. The execution engine
(:mod:`afml.execution.pipeline`) talks only to the abstract contract, so the
mock and the real broker are swappable without touching orchestration code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID, uuid4


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class Order:
    """A market order request.

    ``size`` is in lots / units (the broker decides the contract semantics);
    ``margin`` is the pre-computed margin commitment from the risk engine,
    carried through so the broker can enforce its own margin checks.
    """

    asset: str
    side: OrderSide
    size: float
    margin: float
    order_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class Fill:
    """A realized fill for a submitted order."""

    order_id: UUID
    asset: str
    side: OrderSide
    size: float
    fill_price: float
    margin: float
    status: OrderStatus


@dataclass(frozen=True, slots=True)
class Position:
    """An open position held at the broker."""

    asset: str
    side: OrderSide
    size: float
    entry_price: float
    margin: float


class BrokerAdapter(ABC):
    """Abstract broker contract.

    Implementations must be safe to call from the synchronous execution
    engine. Async brokers should wrap their transport and expose these
    synchronous methods.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish the broker session. Raises on failure."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the session is live."""

    @abstractmethod
    def equity(self) -> float:
        """Current account equity in account currency."""

    @abstractmethod
    def submit_order(self, order: Order, reference_price: float) -> Fill:
        """Submit a market order; return the resulting :class:`Fill`.

        ``reference_price`` is the current mid/last price the engine observed
        — the mock fills at this price; a live broker may fill at its own
        quote and report slippage via the returned ``fill_price``.
        """

    @abstractmethod
    def open_positions(self) -> list[Position]:
        """All currently-open positions."""

    def get_open_positions(self) -> list[Position]:
        """Canonical accessor used by the execution engine's startup
        rehydration (AFML 0-8 final audit V4).

        Defaults to :meth:`open_positions`; live adapters may override to add
        a fresh broker round-trip (rather than returning a cached view) so the
        rehydrated concurrent-position count reflects the broker's true state
        after a restart.
        """
        return self.open_positions()

    @abstractmethod
    def close_position(self, asset: str, reference_price: float) -> Fill | None:
        """Close the open position on ``asset``. Returns the closing fill, or
        ``None`` if there was no open position."""

    @abstractmethod
    def flatten_all(self, reference_prices: dict[str, float]) -> list[Fill]:
        """Close every open position. Returns the list of closing fills.

        ``reference_prices`` maps each open asset to its current price. Assets
        without a price entry are closed at their entry price (no PnL).
        """
