"""Production ASGI entrypoint for the AFML control-plane API (Blueprint §11).

Run with::

    uvicorn apps.api.main:app --host 0.0.0.0 --port 8000

All testable logic lives in :mod:`afml.control_plane` (type-checked + unit
tested under ``tests/unit/phase9``). This module only performs *production
wiring*: it reads :class:`ControlPlaneSettings` from the environment, loads the
CEO TOTP secret from the OS Keychain, opens the Alpha Registry, and stands up a
broker-backed :class:`ExecutionEngine` for the emergency-flatten path.

The CEO **public** key comes from the environment (it is not secret); the
**private** key never touches this server — the CEO signs on their own device.
Until the MetaTrader-5 broker-integration milestone, the flatten path is wired
to the deterministic in-memory mock broker.
"""

from __future__ import annotations

from afml.control_plane import (
    CEOAuthenticator,
    ControlPlaneDeps,
    ControlPlaneSettings,
    create_app,
)
from afml.core.registry.repository import AlphaRegistryRepository
from afml.crypto import get_or_create_ceo_totp_secret
from afml.execution.brokers.mock import InMemoryMockBroker
from afml.execution.pipeline import ExecutionEngine
from afml.execution.risk import RiskEngine


def build_app() -> object:
    """Build the fully-wired production control-plane app."""
    settings = ControlPlaneSettings()
    if not settings.ceo_public_key_hex:
        raise RuntimeError(
            "AFML_CP_CEO_PUBLIC_KEY_HEX is not set — cannot verify CEO signatures"
        )

    # AFML 0-9 polishing audit V3: load the persisted TOTP seed (or create +
    # display it once on first boot) so the CEO's authenticator survives every
    # restart — never a fresh in-memory seed.
    authenticator = CEOAuthenticator(
        public_key_bytes=bytes.fromhex(settings.ceo_public_key_hex),
        totp_secret=get_or_create_ceo_totp_secret(),
    )

    repository = AlphaRegistryRepository(settings.registry_db_url)
    repository.create_all()

    broker = InMemoryMockBroker(starting_equity=settings.account_equity)
    broker.connect()
    risk_engine = RiskEngine(account_equity=settings.account_equity, c95=settings.c95)
    execution_engine = ExecutionEngine(broker=broker, risk_engine=risk_engine)

    deps = ControlPlaneDeps(
        repository=repository,
        execution_engine=execution_engine,
        authenticator=authenticator,
    )
    return create_app(deps, cors_origins=settings.cors_origin_list())


app = build_app()
