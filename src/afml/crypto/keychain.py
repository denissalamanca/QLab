"""macOS Keychain storage for CEO secrets (§11) via the ``keyring`` library.

The CEO's Ed25519 private key and TOTP secret are sensitive material that must
never live in source, env files, or the database. ``keyring`` stores them in
the OS secret store — the macOS Keychain on the lab's dev box, the Secret
Service / Windows Credential Locker elsewhere — encrypted at rest under the
user's login.

This module is a thin, intention-revealing façade so the rest of the codebase
references logical secret names (``ceo_ed25519_private``, ``ceo_totp_secret``)
rather than raw keyring service/username tuples.
"""

from __future__ import annotations

import contextlib

import keyring

# Keyring "service" namespace — keeps AFML secrets distinct from anything else
# in the user's Keychain.
KEYCHAIN_SERVICE: str = "afml-quant-lab"

# Logical secret names (the keyring "username" slot).
CEO_PRIVATE_KEY: str = "ceo_ed25519_private"
CEO_TOTP_SECRET: str = "ceo_totp_secret"


class SecretNotFoundError(KeyError):
    """Raised when a requested secret is absent from the Keychain."""


def store_secret(name: str, value: str, *, service: str = KEYCHAIN_SERVICE) -> None:
    """Persist ``value`` under ``name`` in the OS secret store."""
    keyring.set_password(service, name, value)


def load_secret(name: str, *, service: str = KEYCHAIN_SERVICE) -> str:
    """Retrieve a secret by ``name``. Raises :class:`SecretNotFoundError` if absent."""
    value = keyring.get_password(service, name)
    if value is None:
        raise SecretNotFoundError(f"secret {name!r} not found in keychain service {service!r}")
    return value


def delete_secret(name: str, *, service: str = KEYCHAIN_SERVICE) -> None:
    """Remove a secret (e.g. on key rotation). No-op if already absent."""
    # Idempotent delete — swallow the "already gone" case.
    with contextlib.suppress(keyring.errors.PasswordDeleteError):
        keyring.delete_password(service, name)


def store_ceo_private_key(private_key_hex: str) -> None:
    """Store the CEO Ed25519 private key (hex) in the Keychain."""
    store_secret(CEO_PRIVATE_KEY, private_key_hex)


def store_ceo_totp_secret(totp_secret: str) -> None:
    """Store the CEO TOTP base32 secret in the Keychain."""
    store_secret(CEO_TOTP_SECRET, totp_secret)
