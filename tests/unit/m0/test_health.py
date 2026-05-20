"""M0 — health-check primitives, OOS path config, and the control-plane 403 DoD."""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from afml.config.assets import OOS_SUBDIR, get_asset
from afml.control_plane import CEOAuthenticator, ControlPlaneDeps, create_app
from afml.core.registry.repository import AlphaRegistryRepository
from afml.crypto import (
    CEO_PRIVATE_KEY,
    CEO_TOTP_SECRET,
    generate_keypair,
    generate_totp_secret,
    store_ceo_private_key,
    store_ceo_totp_secret,
)
from afml.execution.brokers.mock import InMemoryMockBroker
from afml.execution.pipeline import ExecutionEngine
from afml.execution.risk import RiskEngine
from afml.ops import (
    Check,
    HealthReport,
    check_files_present,
    check_keychain,
    check_warmup,
)

pytestmark = pytest.mark.m0


# ----------------------------------------------------------- file presence
def test_check_files_present(tmp_path: Path) -> None:
    present = tmp_path / "a.parquet"
    present.write_bytes(b"x")
    assert check_files_present("data", [present]).ok is True
    assert check_files_present("data", [present, tmp_path / "missing.parquet"]).ok is False


# ----------------------------------------------------------------- warm-up
def test_check_warmup_thresholds(tmp_path: Path) -> None:
    big = tmp_path / "big.parquet"
    pl.DataFrame({"x": list(range(100))}).write_parquet(big)
    assert check_warmup("warm-up", [big], min_ticks=50).ok is True
    assert check_warmup("warm-up", [big], min_ticks=500).ok is False
    # Absent file → fail, not crash.
    assert check_warmup("warm-up", [tmp_path / "nope.parquet"], min_ticks=1).ok is False


# ---------------------------------------------------------------- keychain
def test_check_keychain(fake_keyring: dict[tuple[str, str], str]) -> None:
    assert check_keychain("kc", [CEO_PRIVATE_KEY, CEO_TOTP_SECRET]).ok is False
    store_ceo_private_key("deadbeef")
    store_ceo_totp_secret("JBSWY3DPEHPK3PXP")
    assert check_keychain("kc", [CEO_PRIVATE_KEY, CEO_TOTP_SECRET]).ok is True


def test_health_report_aggregation() -> None:
    ok = HealthReport((Check("a", True, ""), Check("b", True, "")))
    bad = HealthReport((Check("a", True, ""), Check("b", False, "x")))
    assert ok.healthy is True and ok.failures() == []
    assert bad.healthy is False and len(bad.failures()) == 1


# ------------------------------------------------------------- OOS config
def test_oos_data_path_format() -> None:
    p = get_asset("EURUSD").oos_data_path
    assert p.parent.name == OOS_SUBDIR
    assert p.name == "EURUSD_2026-01-01_2026-04-30_DUKASCOPY.parquet"


# ------------------------------------------------- control-plane 403 (M0 DoD)
def test_approve_unauthorized_returns_403_not_500(tmp_db_url: str) -> None:
    """A well-formed but unsigned approve on a real strategy → 403 (never 500)."""
    _, public_key = generate_keypair()
    auth = CEOAuthenticator(public_key_bytes=public_key, totp_secret=generate_totp_secret())
    repo = AlphaRegistryRepository(tmp_db_url)
    repo.create_all()
    experiment_id = repo.record_experiment(
        agent_version="m0-test",
        asset="EURUSD",
        algorithmic_family="cusum",
        hyperparameter_vector={"span": 50},
        num_events_triggered=600,
    )
    repo.record_validation(experiment_id, pbo=0.03, dsr=1.4)
    broker = InMemoryMockBroker(starting_equity=100_000.0)
    broker.connect()
    deps = ControlPlaneDeps(
        repository=repo,
        execution_engine=ExecutionEngine(
            broker=broker, risk_engine=RiskEngine(account_equity=100_000.0)
        ),
        authenticator=auth,
    )
    client = TestClient(create_app(deps))

    resp = client.post(
        "/api/v1/execution/approve",
        json={
            "experiment_id": str(experiment_id),
            "timestamp_ms": int(time.time() * 1000),
            "signed_token": "00" * 64,
            "totp_code": "000000",
        },
    )
    assert resp.status_code == 403
    exp = repo.get(experiment_id)
    assert exp is not None and exp.is_deployed is False
