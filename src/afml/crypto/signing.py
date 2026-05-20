"""Ed25519 signing for CEO Human-in-the-Loop capital authorisation (§11).

Every capital-allocating action (Paper → Live promotion, emergency flatten)
must carry a cryptographic signature from the CEO's Ed25519 private key over a
**canonical message** describing the action. The control-plane API verifies
that signature against the registered public key before doing anything. An
unsigned or mis-signed request is rejected — no exceptions.

Ed25519 (RFC 8032) is chosen for short keys/signatures, deterministic
signing, and resistance to the nonce-reuse failures that plague ECDSA.

The private key lives in the macOS Keychain (:mod:`afml.crypto.keychain`); it
never touches disk or source. This module only deals in key *material* +
sign / verify primitives — storage is a separate concern.
"""

from __future__ import annotations

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate a fresh Ed25519 keypair.

    Returns
    -------
    ``(private_key_bytes, public_key_bytes)`` — both raw 32-byte encodings,
    hex-friendly for Keychain storage and config.
    """
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )
    return private_bytes, public_bytes


def public_key_from_private(private_key_bytes: bytes) -> bytes:
    """Derive the raw public key from a raw private key."""
    private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    return private_key.public_key().public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )


def sign_message(private_key_bytes: bytes, message: str) -> str:
    """Sign a canonical message; return the hex-encoded signature.

    Parameters
    ----------
    private_key_bytes
        Raw 32-byte Ed25519 private key.
    message
        The canonical action string (e.g. ``"approve:<experiment_id>"``).
        Encoded as UTF-8 before signing.

    Returns
    -------
    Hex-encoded 64-byte signature.
    """
    private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    signature = private_key.sign(message.encode("utf-8"))
    return signature.hex()


def verify_signature(public_key_bytes: bytes, message: str, signature_hex: str) -> bool:
    """Verify a hex signature over ``message`` against the public key.

    Returns ``True`` iff the signature is valid. Any malformed input
    (bad hex, wrong length, tampered message) returns ``False`` — never
    raises — so the API layer can branch cleanly on the boolean.
    """
    try:
        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        signature = bytes.fromhex(signature_hex)
        public_key.verify(signature, message.encode("utf-8"))
    except (InvalidSignature, ValueError):
        return False
    return True


def approval_message(experiment_id: str, timestamp_ms: int) -> str:
    """Canonical message signed to promote a strategy Paper → Live.

    AFML 0-9 final audit V1: the signature binds the ``experiment_id`` to a
    millisecond UTC ``timestamp_ms``. The control plane rejects a signature
    whose timestamp is outside a ±60 s window, so a captured payload cannot be
    replayed beyond that window (anti-replay).
    """
    return f"afml:approve:{experiment_id}:{timestamp_ms}"


def flatten_message(nonce: str, timestamp_ms: int) -> str:
    """Canonical message signed to trigger an emergency flatten.

    AFML 0-9 final audit V1: the signature binds a caller-supplied ``nonce``
    (single-use replay guard) **and** a millisecond UTC ``timestamp_ms`` (±60 s
    freshness window) — defence in depth so a captured flatten signature is
    neither replayable within the window (nonce already consumed) nor after it
    (timestamp stale), even across a control-plane restart.
    """
    return f"afml:flatten:{nonce}:{timestamp_ms}"
