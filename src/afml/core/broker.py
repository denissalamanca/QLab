"""Asynchronous Redis Pub/Sub broker for inter-agent message passing.

Wraps ``redis.asyncio`` with typed Pydantic ``Event`` envelopes. Supports either a
real Redis instance (production / benchmark) or in-memory ``fakeredis`` (unit tests),
selected by the ``AFML_USE_FAKE_REDIS`` setting.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import orjson
import redis.asyncio as aioredis
from pydantic import TypeAdapter

from afml.config.settings import get_settings
from afml.core.events import Channel, Event, EventEnvelope

if TYPE_CHECKING:
    from redis.asyncio.client import PubSub

logger = logging.getLogger(__name__)
_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)

# AFML 0-8 final audit V3: the inter-agent bus carries ML artifacts —
# Feature matrices (Phase 3), probability arrays (Phase 5), sized bets
# (Phase 7) — that are heavy with ``numpy.float64`` / ``numpy.int64`` /
# ``numpy.ndarray`` and ``pandas.Timestamp``. The stdlib ``json`` module
# (and bare ``orjson``) cannot serialize these and would crash the broker
# the moment one slips through a loosely-typed payload. ``encode_json`` /
# ``decode_json`` below are the canonical, numpy/pandas-safe codec for the
# whole messaging layer.
_ORJSON_OPTS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS


def _json_default(obj: Any) -> Any:
    """Fallback encoder for types orjson cannot serialize natively.

    - ``numpy`` scalars → native Python ``int`` / ``float`` / ``bool``.
    - ``numpy`` arrays → lists (belt-and-suspenders; ``OPT_SERIALIZE_NUMPY``
      already handles most arrays, this covers object/edge dtypes).
    - anything date-like (``pandas.Timestamp``, ``datetime``) → ISO-8601 via
      its ``isoformat`` — duck-typed so we don't import pandas at module load.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    isoformat = getattr(obj, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def encode_json(payload: Any) -> bytes:
    """Serialize any agent payload to JSON bytes, numpy/pandas-safe.

    Use this everywhere a message crosses the bus — not just for ``Event``
    envelopes but for raw prediction arrays / feature dicts too.
    """
    return orjson.dumps(payload, default=_json_default, option=_ORJSON_OPTS)


def decode_json(data: bytes | bytearray | memoryview | str) -> Any:
    """Deserialize JSON bytes produced by :func:`encode_json`."""
    return orjson.loads(data)


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
                data = decode_json(message["data"])
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
        # ``model_dump(mode="json")`` coerces typed fields; ``encode_json``'s
        # numpy/pandas ``default`` is the safety net for any value Pydantic
        # left as a numpy scalar / array inside a loosely-typed field
        # (AFML 0-8 final audit V3).
        return encode_json(event.model_dump(mode="json"))
