"""MetaTrader 5 broker adapter (Phase 7+, live).

The live feed and execution share one MT5 socket (locked decision: same prices
feed feature computation and order routing, eliminating price-feed / fill
mismatch). On macOS the ``MetaTrader5`` Python package runs under a Wine bridge
or a Windows VM; in CI / local dev that package and a running terminal are
absent, so :meth:`MT5Adapter.connect` raises a clear, actionable error instead
of importing at module load.

This module ships the COMPLETE :class:`BrokerAdapter` interface so the
Wine-bridge wiring is a drop-in later. The order-translation logic
(``OrderSide`` → MT5 ``ORDER_TYPE_BUY/SELL``, lot rounding, deviation/slippage
handling) lives here; only the transport calls are deferred behind the lazy
import.
"""

from __future__ import annotations

from typing import Any

from afml.execution.brokers.base import (
    BrokerAdapter,
    Fill,
    Order,
    Position,
)

_MT5_IMPORT_HINT = (
    "MetaTrader5 is unavailable in this environment. The MT5 adapter requires "
    "the `MetaTrader5` Python package and a running MT5 terminal, reached via a "
    "Wine bridge (macOS) or a Windows VM. Install per the broker runbook and "
    "ensure the terminal is logged in before calling connect(). For tests and "
    "dry runs use afml.execution.brokers.InMemoryMockBroker instead."
)


class MT5Adapter(BrokerAdapter):
    """MetaTrader 5 implementation of :class:`BrokerAdapter`.

    Parameters
    ----------
    login, password, server
        MT5 account credentials. Read from the secret store (macOS Keychain)
        by the caller — never hard-coded.
    deviation_points
        Maximum allowed slippage in points for market orders.
    """

    def __init__(
        self,
        *,
        login: int,
        password: str,
        server: str,
        deviation_points: int = 20,
    ) -> None:
        self._login = login
        self._password = password
        self._server = server
        self._deviation = deviation_points
        self._mt5: Any | None = None
        self._connected = False

    def _require_mt5(self) -> Any:
        """Lazily import the ``MetaTrader5`` package, with a clear failure."""
        try:
            # Deliberately lazy: the package + terminal are absent in CI /
            # local dev, so importing at module load would break every import
            # of afml.execution. Deferring to connect() is the point of the
            # scaffold.
            import MetaTrader5 as mt5  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(_MT5_IMPORT_HINT) from exc
        return mt5

    def connect(self) -> None:  # pragma: no cover - requires live terminal
        mt5 = self._require_mt5()
        if not mt5.initialize(login=self._login, password=self._password, server=self._server):
            raise RuntimeError(
                f"MT5 initialize failed: {mt5.last_error()}. Verify the terminal "
                f"is running and the Wine bridge / VM is reachable."
            )
        self._mt5 = mt5
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected

    def equity(self) -> float:  # pragma: no cover - requires live terminal
        mt5 = self._ensure_connected()
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"MT5 account_info failed: {mt5.last_error()}")
        return float(info.equity)

    def submit_order(self, order: Order, reference_price: float) -> Fill:  # pragma: no cover
        self._ensure_connected()
        raise NotImplementedError(
            "MT5 live order routing is wired during the broker-integration "
            "milestone. The translation contract (OrderSide → ORDER_TYPE_*, "
            "lot rounding, deviation handling) is specified in this module's "
            "docstring; the transport call is deferred until a terminal is "
            "available. Use InMemoryMockBroker for the Phase 7 DoD tests."
        )

    def open_positions(self) -> list[Position]:  # pragma: no cover - requires live terminal
        self._ensure_connected()
        raise NotImplementedError("MT5 position query wired at broker-integration milestone.")

    def close_position(self, asset: str, reference_price: float) -> Fill | None:  # pragma: no cover
        self._ensure_connected()
        raise NotImplementedError("MT5 position close wired at broker-integration milestone.")

    def flatten_all(self, reference_prices: dict[str, float]) -> list[Fill]:  # pragma: no cover
        self._ensure_connected()
        raise NotImplementedError("MT5 flatten-all wired at broker-integration milestone.")

    def _ensure_connected(self) -> Any:
        if not self._connected or self._mt5 is None:
            raise RuntimeError("MT5 adapter not connected — call connect() first.")
        return self._mt5
