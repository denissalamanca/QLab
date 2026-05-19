"""Cryptographic CEO authorisation primitives (Blueprint §11).

- **Ed25519 signing** (:mod:`afml.crypto.signing`): keypair generation,
  ``sign_message`` / ``verify_signature``, and the canonical
  ``approval_message`` / ``flatten_message`` builders.
- **TOTP** (:mod:`afml.crypto.totp`): RFC 6238 second factor —
  ``generate_totp_secret``, ``provisioning_uri``, ``verify_totp``.
- **Keychain** (:mod:`afml.crypto.keychain`): macOS Keychain storage of the
  CEO private key + TOTP secret via ``keyring``.

Together these gate every capital-allocating control-plane action behind a
valid signature **and** a live TOTP code.
"""

from afml.crypto.keychain import (
    CEO_PRIVATE_KEY,
    CEO_TOTP_SECRET,
    KEYCHAIN_SERVICE,
    SecretNotFoundError,
    delete_secret,
    load_secret,
    store_ceo_private_key,
    store_ceo_totp_secret,
    store_secret,
)
from afml.crypto.signing import (
    approval_message,
    flatten_message,
    generate_keypair,
    public_key_from_private,
    sign_message,
    verify_signature,
)
from afml.crypto.totp import (
    current_totp,
    generate_totp_secret,
    provisioning_uri,
    verify_totp,
)

__all__ = [
    "CEO_PRIVATE_KEY",
    "CEO_TOTP_SECRET",
    "KEYCHAIN_SERVICE",
    "SecretNotFoundError",
    "approval_message",
    "current_totp",
    "delete_secret",
    "flatten_message",
    "generate_keypair",
    "generate_totp_secret",
    "load_secret",
    "provisioning_uri",
    "public_key_from_private",
    "sign_message",
    "store_ceo_private_key",
    "store_ceo_totp_secret",
    "store_secret",
    "verify_signature",
    "verify_totp",
]
