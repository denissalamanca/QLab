"""End-to-end integration test: Phase 9 control plane (AFML 0-9 audit clearance).

Drives the *real* ASGI app over an in-process ASGI transport against a real
tmp-file SQLite Alpha Registry and a mock-broker-backed execution engine,
asserting the two boundary-hardening contracts the Lead Quant flagged:

* **V1 — cryptographic anti-replay.** A correctly-signed approve with a fresh
  timestamp deploys; a stale timestamp (>60 s) is rejected; a captured flatten
  payload cannot be replayed (single-use nonce).
* **V2 — no event-loop blocking.** Many concurrent ``/registry/strategies``
  reads against real (blocking) SQLite I/O all succeed, because the read is
  offloaded via ``run_in_threadpool``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from uuid import UUID, uuid4

import httpx
import pytest

from afml.control_plane import (
    CEOAuthenticator,
    ControlPlaneDeps,
    InMemoryEventPublisher,
    create_app,
)
from afml.core.registry.repository import AlphaRegistryRepository
from afml.crypto import (
    approval_message,
    current_totp,
    flatten_message,
    generate_keypair,
    generate_totp_secret,
    sign_message,
)
from afml.execution.brokers.base import Order, OrderSide
from afml.execution.brokers.mock import InMemoryMockBroker
from afml.execution.pipeline import ExecutionEngine
from afml.execution.risk import RiskEngine

pytestmark = pytest.mark.integration


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class CPBundle:
    transport: httpx.ASGITransport
    private_key: bytes
    totp_secret: str
    repo: AlphaRegistryRepository
    broker: InMemoryMockBroker
    experiment_id: UUID


@pytest.fixture
def bundle(tmp_db_url: str) -> CPBundle:
    private_key, public_key = generate_keypair()
    totp_secret = generate_totp_secret()

    repo = AlphaRegistryRepository(tmp_db_url)
    repo.create_all()
    experiment_id = repo.record_experiment(
        agent_version="agent_6@phase9-integration",
        asset="XAUUSD",
        algorithmic_family="bollinger",
        hyperparameter_vector={"span": 64, "pt_mult": 2.5, "sl_mult": 1.5},
        num_events_triggered=900,
        orthogonality_score=0.08,
        brain_1_recall=0.79,
        brain_2_log_loss=0.38,
    )
    repo.record_validation(experiment_id, pbo=0.02, dsr=1.65)

    broker = InMemoryMockBroker(starting_equity=100_000.0)
    broker.connect()
    broker.submit_order(
        Order(asset="XAUUSD", side=OrderSide.BUY, size=1.0, margin=750.0), reference_price=2400.0
    )
    broker.submit_order(
        Order(asset="ETHUSD", side=OrderSide.SELL, size=0.5, margin=1500.0), reference_price=3000.0
    )
    engine = ExecutionEngine(broker=broker, risk_engine=RiskEngine(account_equity=100_000.0))

    deps = ControlPlaneDeps(
        repository=repo,
        execution_engine=engine,
        authenticator=CEOAuthenticator(public_key_bytes=public_key, totp_secret=totp_secret),
        event_publisher=InMemoryEventPublisher(),
    )
    return CPBundle(
        transport=httpx.ASGITransport(app=create_app(deps)),
        private_key=private_key,
        totp_secret=totp_secret,
        repo=repo,
        broker=broker,
        experiment_id=experiment_id,
    )


async def test_v2_concurrent_reads_against_real_sqlite(bundle: CPBundle) -> None:
    """25 concurrent reads against a real SQLite file all succeed (no loop stall)."""
    async with httpx.AsyncClient(transport=bundle.transport, base_url="http://cp") as client:
        responses = await asyncio.gather(*[
            client.get("/api/v1/registry/strategies") for _ in range(25)
        ])
    assert all(r.status_code == 200 for r in responses)
    assert all(len(r.json()) == 1 for r in responses)
    assert all(r.json()[0]["dsr"] == pytest.approx(1.65) for r in responses)


async def test_v1_valid_approve_then_stale_rejected(bundle: CPBundle) -> None:
    async with httpx.AsyncClient(transport=bundle.transport, base_url="http://cp") as client:
        # Fresh, correctly-signed approve → deploys.
        ts = _now_ms()
        sig = sign_message(bundle.private_key, approval_message(str(bundle.experiment_id), ts))
        ok = await client.post(
            "/api/v1/execution/approve",
            json={
                "experiment_id": str(bundle.experiment_id),
                "timestamp_ms": ts,
                "signed_token": sig,
                "totp_code": current_totp(bundle.totp_secret),
            },
        )
        assert ok.status_code == 200
        assert ok.json()["deployed"] is True
        # Now off the awaiting list.
        listed = await client.get("/api/v1/registry/strategies")
        assert listed.json() == []

        # A second validated strategy, signed 61 s ago → stale → 403 (replay window).
        other = bundle.repo.record_experiment(
            agent_version="agent_6@phase9-integration",
            asset="XAUUSD",
            algorithmic_family="donchian",
            hyperparameter_vector={"span": 80},
            num_events_triggered=600,
        )
        bundle.repo.record_validation(other, pbo=0.04, dsr=1.1)
        stale_ts = _now_ms() - 61_000
        stale_sig = sign_message(bundle.private_key, approval_message(str(other), stale_ts))
        stale = await client.post(
            "/api/v1/execution/approve",
            json={
                "experiment_id": str(other),
                "timestamp_ms": stale_ts,
                "signed_token": stale_sig,
                "totp_code": current_totp(bundle.totp_secret),
            },
        )
        assert stale.status_code == 403
        fetched = bundle.repo.get(other)
        assert fetched is not None
        assert fetched.is_deployed is False


async def test_v1_flatten_closes_then_replay_rejected(bundle: CPBundle) -> None:
    assert len(bundle.broker.open_positions()) == 2
    nonce = str(uuid4())
    ts = _now_ms()
    sig = sign_message(bundle.private_key, flatten_message(nonce, ts))
    payload = {
        "nonce": nonce,
        "timestamp_ms": ts,
        "signed_token": sig,
        "reason": "integration kill-switch",
        "reference_prices": {"XAUUSD": 2399.0, "ETHUSD": 3010.0},
    }
    async with httpx.AsyncClient(transport=bundle.transport, base_url="http://cp") as client:
        first = await client.post("/api/v1/emergency/flatten", json=payload)
        assert first.status_code == 200
        assert first.json()["n_positions_closed"] == 2
        assert bundle.broker.open_positions() == []

        # Replaying the captured payload is rejected (single-use nonce).
        replay = await client.post("/api/v1/emergency/flatten", json=payload)
        assert replay.status_code == 403
