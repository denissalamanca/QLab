"""Phase 9 — Ed25519 signing primitives (Blueprint §11.2)."""

from __future__ import annotations

import pytest

from afml.crypto import (
    approval_message,
    flatten_message,
    generate_keypair,
    public_key_from_private,
    sign_message,
    verify_signature,
)

pytestmark = pytest.mark.phase9


def test_sign_verify_roundtrip() -> None:
    private_key, public_key = generate_keypair()
    message = approval_message("11111111-1111-1111-1111-111111111111")
    signature = sign_message(private_key, message)
    assert verify_signature(public_key, message, signature) is True


def test_verify_rejects_tampered_message() -> None:
    private_key, public_key = generate_keypair()
    signature = sign_message(private_key, approval_message("exp-a"))
    # Same signature, different message → must not verify.
    assert verify_signature(public_key, approval_message("exp-b"), signature) is False


def test_verify_rejects_wrong_public_key() -> None:
    private_key, _ = generate_keypair()
    _, other_public = generate_keypair()
    signature = sign_message(private_key, "afml:approve:x")
    assert verify_signature(other_public, "afml:approve:x", signature) is False


def test_verify_rejects_malformed_signature() -> None:
    _, public_key = generate_keypair()
    assert verify_signature(public_key, "m", "not-hex-at-all") is False
    assert verify_signature(public_key, "m", "") is False
    # Valid hex but wrong length / wrong bytes.
    assert verify_signature(public_key, "m", "00" * 64) is False


def test_public_key_from_private_matches_generation() -> None:
    private_key, public_key = generate_keypair()
    assert public_key_from_private(private_key) == public_key


def test_canonical_message_format() -> None:
    assert approval_message("e1") == "afml:approve:e1"
    assert flatten_message("n1") == "afml:flatten:n1"


def test_signatures_are_deterministic() -> None:
    # Ed25519 (RFC 8032) signing is deterministic — same key + message ⇒ same sig.
    private_key, _ = generate_keypair()
    message = flatten_message("nonce-42")
    assert sign_message(private_key, message) == sign_message(private_key, message)
