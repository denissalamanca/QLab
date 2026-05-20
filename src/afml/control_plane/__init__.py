"""Phase 9 Control Plane — CEO Human-in-the-Loop governance API (Blueprint §11).

The FastAPI backend that the React frontend drives. It exposes the §11.1
endpoints — list strategies awaiting sign-off, cryptographically approve a
Paper → Live promotion (Ed25519 + TOTP), and the emergency flatten kill-switch
(Ed25519) — all gated through :class:`CEOAuthenticator`.

Public surface:

- :func:`create_app` — the FastAPI application factory.
- :class:`ControlPlaneDeps` — the injected collaborator bundle.
- :class:`CEOAuthenticator` + auth error types — the verification core.
- :class:`InMemoryEventPublisher` / :class:`EventPublisher` — agent-bus seam.
- :class:`ControlPlaneSettings` — production environment configuration.
"""

from afml.control_plane.app import DEFAULT_CORS_ORIGINS, create_app
from afml.control_plane.config import ControlPlaneSettings
from afml.control_plane.deps import (
    ControlPlaneDeps,
    EventPublisher,
    InMemoryEventPublisher,
    get_deps,
)
from afml.control_plane.security import (
    CEOAuthenticator,
    CEOAuthError,
    InvalidSignatureError,
    InvalidTOTPError,
    ReplayError,
    StaleTimestampError,
)

__all__ = [
    "DEFAULT_CORS_ORIGINS",
    "CEOAuthError",
    "CEOAuthenticator",
    "ControlPlaneDeps",
    "ControlPlaneSettings",
    "EventPublisher",
    "InMemoryEventPublisher",
    "InvalidSignatureError",
    "InvalidTOTPError",
    "ReplayError",
    "StaleTimestampError",
    "create_app",
    "get_deps",
]
