"""FastAPI application factory for the Phase 9 control plane.

``create_app`` builds the ASGI app, wires CORS for the React/Vite frontend,
mounts the §11.1 routes, and (optionally) stashes the injected
:class:`ControlPlaneDeps` on ``app.state`` so the per-request dependency can
read them back.

Tests build their own in-memory deps and pass them straight in; the production
entrypoint (``apps/api/main.py``) builds deps from settings + Keychain + a
broker-backed execution engine.
"""

from __future__ import annotations

from collections.abc import Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from afml.control_plane.deps import ControlPlaneDeps
from afml.control_plane.routes import router

#: Vite dev server origin — the default the React frontend serves from.
DEFAULT_CORS_ORIGINS: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")


def create_app(
    deps: ControlPlaneDeps | None = None,
    *,
    cors_origins: Sequence[str] = DEFAULT_CORS_ORIGINS,
) -> FastAPI:
    """Construct the control-plane FastAPI app.

    Parameters
    ----------
    deps
        Wired collaborators (repository, execution engine, authenticator,
        event publisher). When ``None`` the app still starts, but any endpoint
        that needs deps returns HTTP 503 until they are configured — useful for
        OpenAPI generation and health checks.
    cors_origins
        Allowed browser origins for the frontend.
    """
    app = FastAPI(
        title="AFML Quant Lab — Control Plane",
        description="CEO Human-in-the-Loop governance API (Blueprint §11).",
        version="0.9.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    if deps is not None:
        app.state.deps = deps
    app.include_router(router)
    return app
