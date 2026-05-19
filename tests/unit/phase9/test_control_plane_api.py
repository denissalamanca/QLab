"""Phase 9 DoD — control-plane API end-to-end (Blueprint §11.1).

Asserts the Definition-of-Done:
- ``/execution/approve`` rejects unsigned **or** invalid-TOTP requests (403)
  and only deploys on a valid signature + live TOTP.
- ``/emergency/flatten`` propagates to the Agent-7 mock and closes all
  simulated positions (and refuses an unsigned request).
- ``/registry/strategies`` surfaces CPCV-validated strategies awaiting sign-off.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from afml.core.events import CEOApproval, EmergencyFlatten
from afml.crypto import approval_message, current_totp, flatten_message, sign_message
from tests.unit.phase9.conftest import CPHarness

pytestmark = pytest.mark.phase9


# --------------------------------------------------------------- registry list
def test_list_strategies_returns_awaiting_signoff(harness: CPHarness) -> None:
    resp = harness.client.get("/api/v1/registry/strategies")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    row = data[0]
    assert row["experiment_id"] == str(harness.experiment_id)
    assert row["asset"] == "EURUSD"
    assert row["pbo"] == pytest.approx(0.03)
    assert row["dsr"] == pytest.approx(1.42)
    assert row["is_deployed"] is False


# ------------------------------------------------------------------- approve
def test_approve_rejects_unsigned_request(harness: CPHarness) -> None:
    """A valid-hex but bogus signature (no private key) must be rejected."""
    resp = harness.client.post(
        "/api/v1/execution/approve",
        json={
            "experiment_id": str(harness.experiment_id),
            "signed_token": "00" * 64,
            "totp_code": current_totp(harness.totp_secret),
        },
    )
    assert resp.status_code == 403
    exp = harness.repo.get(harness.experiment_id)
    assert exp is not None and exp.is_deployed is False


def test_approve_rejects_empty_signature(harness: CPHarness) -> None:
    resp = harness.client.post(
        "/api/v1/execution/approve",
        json={
            "experiment_id": str(harness.experiment_id),
            "signed_token": "",
            "totp_code": current_totp(harness.totp_secret),
        },
    )
    assert resp.status_code == 403


def test_approve_rejects_invalid_totp(harness: CPHarness) -> None:
    """Valid signature but wrong TOTP → still rejected (mandatory 2FA)."""
    signature = sign_message(harness.private_key, approval_message(str(harness.experiment_id)))
    resp = harness.client.post(
        "/api/v1/execution/approve",
        json={
            "experiment_id": str(harness.experiment_id),
            "signed_token": signature,
            "totp_code": "000000",
        },
    )
    assert resp.status_code == 403
    exp = harness.repo.get(harness.experiment_id)
    assert exp is not None and exp.is_deployed is False


def test_approve_accepts_valid_signature_and_totp(harness: CPHarness) -> None:
    signature = sign_message(harness.private_key, approval_message(str(harness.experiment_id)))
    resp = harness.client.post(
        "/api/v1/execution/approve",
        json={
            "experiment_id": str(harness.experiment_id),
            "signed_token": signature,
            "totp_code": current_totp(harness.totp_secret),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["deployed"] is True

    # State moved Paper → Live.
    exp = harness.repo.get(harness.experiment_id)
    assert exp is not None and exp.is_deployed is True

    # A CEOApproval event was published to the bus.
    assert any(isinstance(evt, CEOApproval) for evt in harness.publisher.events)

    # No longer awaiting sign-off.
    assert harness.client.get("/api/v1/registry/strategies").json() == []


def test_approve_unknown_experiment_returns_404(harness: CPHarness) -> None:
    unknown = uuid4()
    signature = sign_message(harness.private_key, approval_message(str(unknown)))
    resp = harness.client.post(
        "/api/v1/execution/approve",
        json={
            "experiment_id": str(unknown),
            "signed_token": signature,
            "totp_code": current_totp(harness.totp_secret),
        },
    )
    assert resp.status_code == 404


# ------------------------------------------------------------------- flatten
def test_flatten_closes_all_positions(harness: CPHarness) -> None:
    assert len(harness.broker.open_positions()) == 2

    nonce = "flatten-001"
    signature = sign_message(harness.private_key, flatten_message(nonce))
    resp = harness.client.post(
        "/api/v1/emergency/flatten",
        json={
            "nonce": nonce,
            "signed_token": signature,
            "reason": "max drawdown breach",
            "reference_prices": {"EURUSD": 1.09, "BTCUSD": 61_000.0},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["flattened"] is True
    assert body["n_positions_closed"] == 2
    assert len(body["fills"]) == 2

    # Agent-7 mock fully flattened.
    assert harness.broker.open_positions() == []
    # Risk budget reset.
    assert harness.engine.risk_engine.committed_margin == 0.0
    # EmergencyFlatten event published.
    assert any(isinstance(evt, EmergencyFlatten) for evt in harness.publisher.events)


def test_flatten_rejects_unsigned_request(harness: CPHarness) -> None:
    resp = harness.client.post(
        "/api/v1/emergency/flatten",
        json={"nonce": "x", "signed_token": "00" * 64, "reason": "spoofed"},
    )
    assert resp.status_code == 403
    # Positions untouched — a forged kill-switch must not fire.
    assert len(harness.broker.open_positions()) == 2


def test_flatten_replay_is_rejected(harness: CPHarness) -> None:
    nonce = "flatten-replay"
    signature = sign_message(harness.private_key, flatten_message(nonce))
    body = {"nonce": nonce, "signed_token": signature, "reason": "first call"}

    first = harness.client.post("/api/v1/emergency/flatten", json=body)
    assert first.status_code == 200

    replay = harness.client.post("/api/v1/emergency/flatten", json=body)
    assert replay.status_code == 403


# ------------------------------------------------------------------- health
def test_health_ok(harness: CPHarness) -> None:
    resp = harness.client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
