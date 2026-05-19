"""Phase 9 — TOTP second factor (Blueprint §11.2 mandatory 2FA)."""

from __future__ import annotations

import pytest

from afml.crypto import current_totp, generate_totp_secret, provisioning_uri, verify_totp

pytestmark = pytest.mark.phase9


def test_current_code_verifies() -> None:
    secret = generate_totp_secret()
    assert verify_totp(secret, current_totp(secret)) is True


def test_wrong_code_rejected() -> None:
    secret = generate_totp_secret()
    current = current_totp(secret)
    wrong = "000000" if current != "000000" else "111111"
    # valid_window=0 ⇒ only the live code counts, so a different code is a
    # guaranteed (non-flaky) rejection.
    assert verify_totp(secret, wrong, valid_window=0) is False


def test_empty_and_malformed_rejected() -> None:
    secret = generate_totp_secret()
    assert verify_totp(secret, "") is False
    assert verify_totp(secret, "   ") is False
    assert verify_totp(secret, "abcdef") is False


def test_foreign_secret_code_rejected() -> None:
    secret_a = generate_totp_secret()
    secret_b = generate_totp_secret()
    # A code minted under secret_b must not validate under secret_a.
    assert verify_totp(secret_a, current_totp(secret_b), valid_window=0) is False


def test_provisioning_uri_carries_issuer_and_account() -> None:
    secret = generate_totp_secret()
    uri = provisioning_uri(secret, account_name="ceo@afml")
    assert uri.startswith("otpauth://totp/")
    assert "AFML%20Quant%20Lab" in uri or "AFML+Quant+Lab" in uri
    assert "ceo%40afml" in uri or "ceo@afml" in uri
