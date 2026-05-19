"""Dependency-injection wiring for the control plane.

The API talks to three collaborators, all behind interfaces so tests can
inject in-memory doubles and production can wire real services:

- the :class:`AlphaRegistryRepository` (strategies + deployment flag),
- the :class:`ExecutionEngine` (the Agent-7 seam that flattens positions), and
- the :class:`CEOAuthenticator` (signature + TOTP verification).

Plus an :class:`EventPublisher` for emitting ``CEOApproval`` /
``EmergencyFlatten`` onto the agent bus. The default
:class:`InMemoryEventPublisher` records events so tests can assert propagation
without standing up Redis; production swaps in a Redis-backed publisher.

The bundle is stashed on ``app.state.deps`` by :func:`create_app`; the
:func:`get_deps` FastAPI dependency reads it back per-request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from fastapi import HTTPException, Request, status

from afml.control_plane.security import CEOAuthenticator
from afml.core.events import Event
from afml.core.registry.repository import AlphaRegistryRepository
from afml.execution.pipeline import ExecutionEngine


@runtime_checkable
class EventPublisher(Protocol):
    """Anything that can publish a domain :class:`Event` to the agent bus."""

    def publish(self, event: Event) -> None: ...


@dataclass(slots=True)
class InMemoryEventPublisher:
    """Records published events in memory (default; test-friendly).

    Production replaces this with a Redis-backed publisher that serialises the
    event onto the relevant :class:`afml.core.events.Channel`.
    """

    events: list[Event] = field(default_factory=list)

    def publish(self, event: Event) -> None:
        self.events.append(event)


@dataclass(slots=True)
class ControlPlaneDeps:
    """The collaborators a running control plane needs."""

    repository: AlphaRegistryRepository
    execution_engine: ExecutionEngine
    authenticator: CEOAuthenticator
    event_publisher: EventPublisher = field(default_factory=InMemoryEventPublisher)


def get_deps(request: Request) -> ControlPlaneDeps:
    """FastAPI dependency: pull the wired :class:`ControlPlaneDeps` off app state."""
    deps = getattr(request.app.state, "deps", None)
    if not isinstance(deps, ControlPlaneDeps):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="control plane dependencies not configured",
        )
    return deps
