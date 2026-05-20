"""Phase 7 orchestrator — signals → sized bets → risk → broker dispatch.

The :class:`ExecutionEngine` is the single seam between Brain 2's calibrated
probabilities and the broker. For a batch of signals it:

1. Sizes each bet from its probability (:mod:`afml.execution.bet_sizing`),
   using the Mixture-of-Gaussians fallback when the batch is non-Gaussian.
2. Pushes each size through the :class:`RiskEngine` — concurrent-position
   scaling, ESMA leverage, and the FTMO drawdown-buffer hard cap.
3. Translates the surviving margin commitments into broker orders and
   dispatches them, collecting fills.

The engine never lets total committed margin exceed the FTMO buffer, even
under a pathological burst of simultaneous max-confidence signals — that is
the Blueprint §9.3 margin-constraint guarantee.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np

from afml.config.assets import AssetClass, get_asset
from afml.execution.bet_sizing import bet_sizes_for_batch
from afml.execution.brokers.base import BrokerAdapter, Fill, Order, OrderSide
from afml.execution.risk import RiskEngine, SizedBet


@dataclass(frozen=True, slots=True)
class Signal:
    """A Brain-2 trade signal ready for sizing.

    Attributes
    ----------
    asset
        Symbol from the 14-asset universe.
    probability
        Calibrated ``P(success)`` from Brain 2.
    side
        Direction of the primary (Brain 1) signal.
    reference_price
        Current price used for the (mock) fill.
    """

    asset: str
    probability: float
    side: OrderSide
    reference_price: float


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """Outcome of an :meth:`ExecutionEngine.execute_batch` call."""

    fills: list[Fill]
    sized_bets: list[SizedBet]
    total_margin_committed: float
    n_orders_submitted: int
    n_skipped: int
    used_mixture_fallback: bool

    @property
    def n_signals(self) -> int:
        return len(self.sized_bets)


@dataclass(frozen=True, slots=True)
class RehydrationResult:
    """Outcome of :meth:`ExecutionEngine.rehydrate_state`.

    Attributes
    ----------
    n_open_positions
        Concurrent positions discovered at the broker on startup.
    rehydrated_margin
        Total margin those positions commit — seeded into the risk engine.
    """

    n_open_positions: int
    rehydrated_margin: float


@dataclass(slots=True)
class ExecutionEngine:
    """Sizes and dispatches a batch of Brain-2 signals under risk limits.

    Parameters
    ----------
    broker
        Any :class:`BrokerAdapter` (mock or live MT5). Must be connected.
    risk_engine
        The :class:`RiskEngine` holding the account equity, ``c_95``, and
        FTMO buffer.
    min_margin
        Margin commitments below this (in account currency) are treated as
        skips — sub-threshold bets aren't worth the transaction cost.
    """

    broker: BrokerAdapter
    risk_engine: RiskEngine
    min_margin: float = 1e-9
    _dispatched: list[Fill] = field(default_factory=list)
    # AFML 0-9 final audit V4: serializes the live fetch→size→dispatch sequence
    # so two concurrently-arriving signals can never size against the same
    # stale margin snapshot. Created lazily on first async use so it binds to
    # the running event loop (Agent 7's asyncio runtime).
    _lock: asyncio.Lock | None = field(default=None, repr=False)

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def rehydrate_state(self) -> RehydrationResult:
        """Restore concurrent-position state from the broker on startup.

        AFML 0-8 final audit V4: Agent 7 sizes bets against the FTMO drawdown
        buffer using its running committed-margin total. After a restart /
        crash that total is 0 in memory, but the broker may still hold open
        positions. Querying ``broker.get_open_positions()`` and seeding the
        risk engine with their summed margin prevents the engine from
        over-sizing the next signal and breaching the margin limits because it
        "forgot" the live book.

        Call this once, immediately after ``broker.connect()``, before the
        first :meth:`execute_batch`.
        """
        if not self.broker.is_connected():
            raise RuntimeError("broker not connected — call broker.connect() first")
        positions = self.broker.get_open_positions()
        total_margin = float(sum(p.margin for p in positions))
        self.risk_engine.rehydrate_committed_margin(total_margin)
        return RehydrationResult(
            n_open_positions=len(positions),
            rehydrated_margin=total_margin,
        )

    def execute_batch(self, signals: list[Signal], *, random_state: int = 0) -> DispatchResult:
        """Size, risk-check, and dispatch a batch of signals.

        Bets are processed in descending probability order so the highest-
        confidence signals claim the margin budget first when it's scarce.
        """
        if not self.broker.is_connected():
            raise RuntimeError("broker not connected — call broker.connect() first")
        if not signals:
            return DispatchResult([], [], 0.0, 0, 0, False)

        probs = np.array([s.probability for s in signals], dtype=np.float64)
        batch = bet_sizes_for_batch(probs, random_state=random_state)

        # Process highest-confidence first (descending probability).
        order_indices = np.argsort(-probs, kind="mergesort")

        fills: list[Fill] = []
        sized_bets: list[SizedBet] = []
        n_submitted = 0
        n_skipped = 0

        for idx in order_indices:
            signal = signals[idx]
            raw_size = float(batch.sizes[idx])
            asset_class = self._asset_class(signal.asset)
            sized = self.risk_engine.size_bet(raw_size, asset_class)
            sized_bets.append(sized)

            if sized.margin <= self.min_margin:
                n_skipped += 1
                continue

            order = Order(
                asset=signal.asset,
                side=signal.side,
                size=sized.scaled_size,
                margin=sized.margin,
            )
            fill = self.broker.submit_order(order, signal.reference_price)
            fills.append(fill)
            n_submitted += 1

        self._dispatched.extend(fills)
        return DispatchResult(
            fills=fills,
            sized_bets=sized_bets,
            total_margin_committed=self.risk_engine.committed_margin,
            n_orders_submitted=n_submitted,
            n_skipped=n_skipped,
            used_mixture_fallback=batch.used_mixture_fallback,
        )

    async def execute_batch_async(
        self,
        signals: list[Signal],
        *,
        random_state: int = 0,
        rehydrate_first: bool = True,
    ) -> DispatchResult:
        """Race-free async execution for Agent 7's live tick loop (audit V4).

        In a live asynchronous environment a probability signal can arrive from
        Brain 2 *while* a previous signal is still awaiting the broker's
        open-position / margin response. Both would then size against the same
        stale concurrent-trade count and over-leverage the account. This method
        eliminates that race by holding an :class:`asyncio.Lock` across the
        whole critical section, strictly sequentialising:

            acquire lock → fetch live margin/positions → size → dispatch → release

        No second signal is sized until the first has dispatched and its margin
        is reflected in the broker book. The blocking broker round-trips run in
        worker threads (``asyncio.to_thread``) so the event loop stays
        responsive while the lock guarantees mutual exclusion.
        """
        async with self._get_lock():
            if rehydrate_first:
                # Fetch the *current* live margin/positions inside the lock so
                # the size step below cannot use a stale snapshot.
                await asyncio.to_thread(self.rehydrate_state)
            return await asyncio.to_thread(self.execute_batch, signals, random_state=random_state)

    def emergency_flatten(self, reference_prices: dict[str, float]) -> list[Fill]:
        """Close every open position and reset the risk budget.

        Wired to the Phase 9 ``/emergency/flatten`` control-plane endpoint.
        """
        closes = self.broker.flatten_all(reference_prices)
        self.risk_engine.reset()
        return closes

    async def emergency_flatten_async(self, reference_prices: dict[str, float]) -> list[Fill]:
        """Lock-guarded async flatten (audit V4).

        Acquires the same execution lock as :meth:`execute_batch_async` so a
        kill-switch can never interleave with an in-flight size→dispatch
        sequence in Agent 7's runtime.
        """
        async with self._get_lock():
            return await asyncio.to_thread(self.emergency_flatten, reference_prices)

    @staticmethod
    def _asset_class(symbol: str) -> AssetClass:
        """Resolve the asset class for ESMA leverage. Falls back to FX-tier
        only for symbols inside the universe; unknown symbols raise."""
        return get_asset(symbol).asset_class
