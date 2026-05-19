"""Asynchronous Redis Pub/Sub broker for inter-agent message passing.

Wraps ``redis.asyncio`` with typed Pydantic ``Event`` envelopes. Supports either a
real Redis instance (production / benchmark) or in-memory ``fakeredis`` (unit tests),
selected by the ``AFML_USE_FAKE_REDIS`` setting.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

import orjson
import redis.asyncio as aioredis
from pydantic import TypeAdapter

from afml.config.settings import get_settings
from afml.core.events import Channel, Event, EventEnvelope

if TYPE_CHECKING:
    from redis.asyncio.client import PubSub

logger = logging.getLogger(__name__)
_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)


class MessageBroker:
    """Thin async wrapper over Redis Pub/Sub with typed envelopes.

    Designed for non-blocking, high-throughput inter-agent messaging. The
    Definition-of-Done (Blueprint §2.3) requires ≥ 10,000 JSON payloads/sec
    end-to-end with zero dropped packets.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis: aioredis.Redis = redis_client
        self._pubsub: PubSub | None = None

    @classmethod
    async def connect(cls, url: str | None = None) -> MessageBroker:
        """Open a connection. URL defaults to ``settings.redis_url``.

        If ``settings.use_fake_redis`` is true, returns an in-memory fake instead.
        """
        settings = get_settings()
        client: aioredis.Redis
        if settings.use_fake_redis:
            # local import — fakeredis is a dev-only dep, not pulled into prod
            import fakeredis.aioredis as fake  # noqa: PLC0415

            client = fake.FakeRedis(decode_responses=False)
        else:
            client = aioredis.from_url(
                url or settings.redis_url,
                encoding="utf-8",
                decode_responses=False,
                socket_connect_timeout=5,
            )
            await client.ping()  # type: ignore[misc]
        return cls(client)

    async def close(self) -> None:
        if self._pubsub is not None:
            try:
                await self._pubsub.aclose()  # type: ignore[no-untyped-call]
            except Exception:
                logger.debug("pubsub close suppressed", exc_info=True)
            self._pubsub = None
        await self._redis.aclose()

    # ----------------------------------------------------------------- publish
    async def publish(self, event: EventEnvelope) -> int:
        """Publish one event to its declared channel. Returns # subscribers reached."""
        payload = self._encode(event)
        return int(await self._redis.publish(event.channel.value, payload))

    async def publish_many(self, events: Sequence[EventEnvelope]) -> int:
        """Pipeline-publish a batch. Returns the cumulative subscriber count."""
        if not events:
            return 0
        async with self._redis.pipeline(transaction=False) as pipe:
            for event in events:
                pipe.publish(event.channel.value, self._encode(event))
            results: list[Any] = await pipe.execute()
        return sum(int(r) for r in results)

    # --------------------------------------------------------------- subscribe
    async def subscribe(self, *channels: Channel) -> AsyncIterator[Event]:
        """Async iterator yielding deserialized ``Event``s from given channels."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(*[c.value for c in channels])
        self._pubsub = pubsub
        try:
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                data = orjson.loads(message["data"])
                yield _event_adapter.validate_python(data)
        finally:
            await pubsub.unsubscribe(*[c.value for c in channels])

    async def consume(
        self,
        channels: list[Channel],
        handler: Callable[[Event], Awaitable[None]],
    ) -> None:
        """Drive a long-running consumer. Catches per-message handler errors."""
        async for event in self.subscribe(*channels):
            try:
                await handler(event)
            except Exception:
                logger.exception("handler failed for event %s", event.event_id)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _encode(event: EventEnvelope) -> bytes:
        return orjson.dumps(event.model_dump(mode="json"))
