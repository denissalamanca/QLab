"""In-memory mock broker — deterministic simulator for Phases 0-7.

Fills every accepted order at the supplied reference price, tracks open
positions and committed margin, and rejects orders that would exceed the
account's margin budget. No network, no randomness — fully reproducible.
"""

from __future__ import annotations

from afml.execution.brokers.base import (
    BrokerAdapter,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)


class InMemoryMockBroker(BrokerAdapter):
    """Deterministic broker simulator.

    Parameters
    ----------
    starting_equity
        Account equity at construction.
    margin_budget
        Optional hard ceiling on total committed margin. Orders that would
        push committed margin beyond this are ``REJECTED``. ``None`` ⇒ no
        broker-side margin ceiling (the risk engine is then the sole guard).
    """

    def __init__(self, starting_equity: float, *, margin_budget: float | None = None) -> None:
        if starting_equity <= 0:
            raise ValueError(f"starting_equity must be > 0, got {starting_equity}")
        self._equity = float(starting_equity)
        self._margin_budget = margin_budget
        self._connected = False
        self._positions: dict[str, Position] = {}
        self._committed_margin = 0.0
        self.fills: list[Fill] = []

    # ------------------------------------------------------------- connection
    def connect(self) -> None:
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected

    def equity(self) -> float:
        return self._equity

    @property
    def committed_margin(self) -> float:
        return self._committed_margin

    # ------------------------------------------------------------------ orders
    def submit_order(self, order: Order, reference_price: float) -> Fill:
        if not self._connected:
            raise RuntimeError("broker not connected — call connect() first")
        if order.size <= 0.0 or order.margin <= 0.0:
            # Zero / negative size is a no-op rejection (e.g. p ≤ 0.5 bets).
            fill = Fill(
                order_id=order.order_id,
                asset=order.asset,
                side=order.side,
                size=order.size,
                fill_price=reference_price,
                margin=0.0,
                status=OrderStatus.REJECTED,
            )
            self.fills.append(fill)
            return fill

        # Broker-side margin ceiling.
        if (
            self._margin_budget is not None
            and self._committed_margin + order.margin > self._margin_budget + 1e-9
        ):
            fill = Fill(
                order_id=order.order_id,
                asset=order.asset,
                side=order.side,
                size=order.size,
                fill_price=reference_price,
                margin=0.0,
                status=OrderStatus.REJECTED,
            )
            self.fills.append(fill)
            return fill

        self._positions[order.asset] = Position(
            asset=order.asset,
            side=order.side,
            size=order.size,
            entry_price=reference_price,
            margin=order.margin,
        )
        self._committed_margin += order.margin
        fill = Fill(
            order_id=order.order_id,
            asset=order.asset,
            side=order.side,
            size=order.size,
            fill_price=reference_price,
            margin=order.margin,
            status=OrderStatus.FILLED,
        )
        self.fills.append(fill)
        return fill

    def open_positions(self) -> list[Position]:
        return list(self._positions.values())

    def close_position(self, asset: str, reference_price: float) -> Fill | None:
        pos = self._positions.pop(asset, None)
        if pos is None:
            return None
        self._committed_margin -= pos.margin
        # Realize PnL into equity (long: +Δ, short: −Δ).
        direction = 1.0 if pos.side == OrderSide.BUY else -1.0
        pnl = direction * (reference_price - pos.entry_price) * pos.size
        self._equity += pnl
        fill = Fill(
            order_id=Order(pos.asset, pos.side, pos.size, pos.margin).order_id,
            asset=pos.asset,
            side=OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY,
            size=pos.size,
            fill_price=reference_price,
            margin=pos.margin,
            status=OrderStatus.CLOSED,
        )
        self.fills.append(fill)
        return fill

    def flatten_all(self, reference_prices: dict[str, float]) -> list[Fill]:
        closes: list[Fill] = []
        for asset in list(self._positions.keys()):
            price = reference_prices.get(asset, self._positions[asset].entry_price)
            fill = self.close_position(asset, price)
            if fill is not None:
                closes.append(fill)
        return closes
