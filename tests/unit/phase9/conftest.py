"""Phase 9 fixtures — a fully in-memory control-plane harness.

Builds a :class:`fastapi.testclient.TestClient` over :func:`create_app` wired
with: a known Ed25519 keypair + TOTP secret (so tests can forge *valid*
signatures), an Alpha Registry seeded with one CPCV-validated strategy awaiting
sign-off, and a mock broker holding two open positions for the flatten path.
No Keychain, no Redis, no real broker.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from afml.control_plane import (
    CEOAuthenticator,
    ControlPlaneDeps,
    InMemoryEventPublisher,
    create_app,
)
from afml.core.registry.repository import AlphaRegistryRepository
from afml.crypto import generate_keypair, generate_totp_secret
from afml.execution.brokers.base import Order, OrderSide
from afml.execution.brokers.mock import InMemoryMockBroker
from afml.execution.pipeline import ExecutionEngine
from afml.execution.risk import RiskEngine


@dataclass
class CPHarness:
    """Bundle of everything a control-plane test needs to drive + assert."""

    app: FastAPI
    client: TestClient
    private_key: bytes
    public_key: bytes
    totp_secret: str
    repo: AlphaRegistryRepository
    broker: InMemoryMockBroker
    engine: ExecutionEngine
    publisher: InMemoryEventPublisher
    experiment_id: UUID


@pytest.fixture
def harness(tmp_db_url: str) -> CPHarness:
    private_key, public_key = generate_keypair()
    totp_secret = generate_totp_secret()
    authenticator = CEOAuthenticator(public_key_bytes=public_key, totp_secret=totp_secret)

    repo = AlphaRegistryRepository(tmp_db_url)
    repo.create_all()
    experiment_id = repo.record_experiment(
        agent_version="agent_6@phase9-test",
        asset="EURUSD",
        algorithmic_family="cusum",
        hyperparameter_vector={"span": 50, "pt_mult": 2.0, "sl_mult": 2.0},
        num_events_triggered=750,
        orthogonality_score=0.12,
        brain_1_recall=0.82,
        brain_2_log_loss=0.41,
    )
    # Phase 6 validation outcomes → makes it surface on /registry/strategies.
    repo.record_validation(experiment_id, pbo=0.03, dsr=1.42)

    broker = InMemoryMockBroker(starting_equity=100_000.0)
    broker.connect()
    # Two live positions for the emergency-flatten path.
    broker.submit_order(
        Order(asset="EURUSD", side=OrderSide.BUY, size=1.0, margin=500.0),
        reference_price=1.10,
    )
    broker.submit_order(
        Order(asset="BTCUSD", side=OrderSide.SELL, size=0.5, margin=2_000.0),
        reference_price=60_000.0,
    )
    engine = ExecutionEngine(
        broker=broker,
        risk_engine=RiskEngine(account_equity=100_000.0),
    )

    publisher = InMemoryEventPublisher()
    deps = ControlPlaneDeps(
        repository=repo,
        execution_engine=engine,
        authenticator=authenticator,
        event_publisher=publisher,
    )
    app = create_app(deps)
    client = TestClient(app)
    return CPHarness(
        app=app,
        client=client,
        private_key=private_key,
        public_key=public_key,
        totp_secret=totp_secret,
        repo=repo,
        broker=broker,
        engine=engine,
        publisher=publisher,
        experiment_id=experiment_id,
    )
