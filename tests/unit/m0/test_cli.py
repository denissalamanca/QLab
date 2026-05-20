"""M0 — `afml` CLI: enroll-ceo + doctor (Ops roadmap M0 DoD)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from afml.cli import app
from afml.crypto import CEO_PRIVATE_KEY, CEO_TOTP_SECRET, KEYCHAIN_SERVICE
from afml.ops import Check, HealthReport

pytestmark = pytest.mark.m0

runner = CliRunner()


# ---------------------------------------------------------------- enroll-ceo
def test_enroll_fresh_persists_and_prints(fake_keyring: dict[tuple[str, str], str]) -> None:
    result = runner.invoke(app, ["enroll-ceo"])
    assert result.exit_code == 0, result.output
    assert "CEO public key" in result.output
    assert "otpauth://" in result.output
    # Both secrets persisted to the (fake) Keychain.
    assert (KEYCHAIN_SERVICE, CEO_PRIVATE_KEY) in fake_keyring
    assert (KEYCHAIN_SERVICE, CEO_TOTP_SECRET) in fake_keyring


def test_enroll_is_idempotent_without_force(fake_keyring: dict[tuple[str, str], str]) -> None:
    first = runner.invoke(app, ["enroll-ceo"])
    priv_after_first = fake_keyring[(KEYCHAIN_SERVICE, CEO_PRIVATE_KEY)]

    second = runner.invoke(app, ["enroll-ceo"])
    assert second.exit_code == 0
    assert "already enrolled" in second.output.lower()
    # Key NOT rotated.
    assert fake_keyring[(KEYCHAIN_SERVICE, CEO_PRIVATE_KEY)] == priv_after_first
    # Same public key reported both times.
    assert first.exit_code == 0


def test_enroll_force_rotates_keys(fake_keyring: dict[tuple[str, str], str]) -> None:
    runner.invoke(app, ["enroll-ceo"])
    priv_before = fake_keyring[(KEYCHAIN_SERVICE, CEO_PRIVATE_KEY)]

    result = runner.invoke(app, ["enroll-ceo", "--force"])
    assert result.exit_code == 0
    assert fake_keyring[(KEYCHAIN_SERVICE, CEO_PRIVATE_KEY)] != priv_before


# ------------------------------------------------------------------- doctor
def test_doctor_exits_zero_when_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    healthy = HealthReport((Check("data", True, "ok"), Check("redis", True, "ok")))
    monkeypatch.setattr("afml.cli.run_health_checks", lambda: healthy)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "All systems healthy" in result.output


def test_doctor_exits_one_with_specific_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    unhealthy = HealthReport((
        Check("data", True, "ok"),
        Check("redis bus", False, "unreachable: refused"),
    ))
    monkeypatch.setattr("afml.cli.run_health_checks", lambda: unhealthy)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "redis bus" in result.output
