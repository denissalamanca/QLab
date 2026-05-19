"""Phase 0 — Redis Pub/Sub broker.

Tests use the in-memory ``fakeredis`` backend (forced by ``conftest.py``). The
real-Redis 10 000 msg/sec throughput benchmark lives in
``tests/benchmarks/test_broker_throughput.py`` and runs only with
``--run-benchmarks``.
"""

from __future__ import annotations

import asyncio
from typing import cast
from uuid import uuid4

import pytest

from afml.core.broker import MessageBroker
from afml.core.events import (
    BarGenerated,
    Channel,
    Event,
    StrategyValidated,
)


@pytest.mark.phase0
@pytest.mark.asyncio
async def test_broker_publish_subscribe_roundtrip() -> None:
    broker = await MessageBroker.connect()
    try:
        received: list[Event] = []

        async def collect() -> None:
            async for ev in broker.subscribe(Channel.BAR_GENERATED):
                received.append(ev)
                if len(received) == 3:
                    return

        consumer = asyncio.create_task(collect())
        # Allow the subscription to register before we start publishing.
        await asyncio.sleep(0.05)

        for i in range(3):
            await broker.publish(
                BarGenerated(
                    producer="agent_1",
                    asset="EURUSD",
                    bar_type="time",
                    bar_count=i,
                    jarque_bera_stat=float(i),
                )
            )

        await asyncio.wait_for(consumer, timeout=2.0)
        assert len(received) == 3
        for r in received:
            assert isinstance(r, BarGenerated)
    finally:
        await broker.close()


@pytest.mark.phase0
@pytest.mark.asyncio
async def test_broker_publish_many_accepts_batch() -> None:
    broker = await MessageBroker.connect()
    try:
        events = [
            BarGenerated(
                producer="agent_1",
                asset="EURUSD",
                bar_type="time",
                bar_count=i,
                jarque_bera_stat=1.0,
            )
            for i in range(50)
        ]
        # No subscriber → each publish returns 0, total 0. We're verifying the
        # pipeline path accepts and dispatches the batch without raising.
        result = await broker.publish_many(list(events))
        assert result == 0
    finally:
        await broker.close()


@pytest.mark.phase0
@pytest.mark.asyncio
async def test_broker_typed_roundtrip_preserves_payload() -> None:
    """The discriminated-union deserializer must rebuild the exact concrete type."""
    broker = await MessageBroker.connect()
    try:
        received: list[Event] = []

        async def collect() -> None:
            async for ev in broker.subscribe(Channel.STRATEGY_VALIDATED):
                received.append(ev)
                return

        consumer = asyncio.create_task(collect())
        await asyncio.sleep(0.05)

        exp_id = uuid4()
        await broker.publish(
            StrategyValidated(
                producer="agent_6",
                asset="EURUSD",
                experiment_id=exp_id,
                pbo=0.03,
                dsr=1.4,
                fwer_penalty=0.21,
                approved=True,
            )
        )

        await asyncio.wait_for(consumer, timeout=2.0)
        assert len(received) == 1
        sv = cast(StrategyValidated, received[0])
        assert isinstance(sv, StrategyValidated)
        assert sv.experiment_id == exp_id
        assert sv.dsr == pytest.approx(1.4)
        assert sv.approved is True
    finally:
        await broker.close()


@pytest.mark.phase0
@pytest.mark.asyncio
async def test_broker_publish_many_empty_is_noop() -> None:
    broker = await MessageBroker.connect()
    try:
        assert await broker.publish_many([]) == 0
    finally:
        await broker.close()
