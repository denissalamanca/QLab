"""Phase 9 — CEOAuthenticator verification core (Blueprint §11.2 + audit V1).

Unit-level coverage of the security seam the API delegates to: approval needs
signature **and** TOTP; flatten needs signature only + a per-nonce replay
guard; both bind a fresh ±60 s timestamp to the signature (anti-replay).
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

from afml.control_plane import (
    CEOAuthenticator,
    InvalidSignatureError,
    InvalidTOTPError,
    ReplayError,
    StaleTimestampError,
)
from afml.crypto import (
    approval_message,
    current_totp,
    flatten_message,
    generate_keypair,
    generate_totp_secret,
    sign_message,
)

pytestmark = pytest.mark.phase9


def _now_ms() -> int:
    return int(time.time() * 1000)


def _make_authenticator(*, totp_valid_window: int = 1) -> tuple[CEOAuthenticator, bytes, str]:
    private_key, public_key = generate_keypair()
    totp_secret = generate_totp_secret()
    auth = CEOAuthenticator(
        public_key_bytes=public_key,
        totp_secret=totp_secret,
        totp_valid_window=totp_valid_window,
    )
    return auth, private_key, totp_secret


def test_verify_approval_accepts_valid_signature_and_totp() -> None:
    auth, private_key, totp_secret = _make_authenticator()
    experiment_id = uuid4()
    ts = _now_ms()
    signature = sign_message(private_key, approval_message(str(experiment_id), ts))
    # Must not raise.
    auth.verify_approval(experiment_id, ts, signature, current_totp(totp_secret))


def test_verify_approval_rejects_bad_signature() -> None:
    auth, _, totp_secret = _make_authenticator()
    experiment_id = uuid4()
    with pytest.raises(InvalidSignatureError):
        auth.verify_approval(experiment_id, _now_ms(), "00" * 64, current_totp(totp_secret))


def test_verify_approval_rejects_bad_totp() -> None:
    auth, private_key, totp_secret = _make_authenticator(totp_valid_window=0)
    experiment_id = uuid4()
    ts = _now_ms()
    signature = sign_message(private_key, approval_message(str(experiment_id), ts))
    current = current_totp(totp_secret)
    wrong = "000000" if current != "000000" else "111111"
    with pytest.raises(InvalidTOTPError):
        auth.verify_approval(experiment_id, ts, signature, wrong)


def test_verify_approval_rejects_stale_timestamp() -> None:
    """Audit V1: a signature minted >60 s ago is rejected (replay window)."""
    auth, private_key, totp_secret = _make_authenticator()
    experiment_id = uuid4()
    stale_ts = _now_ms() - 61_000  # 61 s in the past
    signature = sign_message(private_key, approval_message(str(experiment_id), stale_ts))
    with pytest.raises(StaleTimestampError):
        auth.verify_approval(experiment_id, stale_ts, signature, current_totp(totp_secret))


def test_verify_approval_rejects_future_timestamp() -> None:
    auth, private_key, totp_secret = _make_authenticator()
    experiment_id = uuid4()
    future_ts = _now_ms() + 120_000  # 2 min in the future
    signature = sign_message(private_key, approval_message(str(experiment_id), future_ts))
    with pytest.raises(StaleTimestampError):
        auth.verify_approval(experiment_id, future_ts, signature, current_totp(totp_secret))


def test_verify_flatten_accepts_valid_then_blocks_replay() -> None:
    auth, private_key, _ = _make_authenticator()
    nonce = "flatten-nonce-1"
    ts = _now_ms()
    signature = sign_message(private_key, flatten_message(nonce, ts))
    auth.verify_flatten(nonce, ts, signature)  # first use ok
    with pytest.raises(ReplayError):
        auth.verify_flatten(nonce, ts, signature)  # replay rejected


def test_verify_flatten_rejects_bad_signature() -> None:
    auth, _, _ = _make_authenticator()
    with pytest.raises(InvalidSignatureError):
        auth.verify_flatten("flatten-nonce-2", _now_ms(), "00" * 64)


def test_verify_flatten_rejects_empty_nonce() -> None:
    auth, private_key, _ = _make_authenticator()
    ts = _now_ms()
    signature = sign_message(private_key, flatten_message("", ts))
    with pytest.raises(InvalidSignatureError):
        auth.verify_flatten("", ts, signature)


def test_verify_flatten_rejects_stale_timestamp() -> None:
    """Audit V1: a captured flatten signature is unusable once its timestamp ages out."""
    auth, private_key, _ = _make_authenticator()
    nonce = "flatten-stale"
    stale_ts = _now_ms() - 61_000
    signature = sign_message(private_key, flatten_message(nonce, stale_ts))
    with pytest.raises(StaleTimestampError):
        auth.verify_flatten(nonce, stale_ts, signature)
