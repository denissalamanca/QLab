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
from collections.abc import Callable

import keyring

from afml.crypto.totp import generate_totp_secret, provisioning_uri

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


def get_or_create_ceo_totp_secret(
    *,
    account_name: str = "ceo@afml-quant-lab",
    issuer: str = "AFML Quant Lab",
    echo: Callable[[str], None] = print,
) -> str:
    """Return the persisted CEO TOTP secret, creating it once on first run.

    AFML 0-9 polishing audit V3 — *ephemeral seed lockout*. A TOTP seed
    generated fresh in memory on each control-plane boot would invalidate the
    seed the CEO scanned into their authenticator app on the very first restart,
    permanently locking them out of ``/execution/approve`` and
    ``/emergency/flatten``. This get-or-create makes the seed **deterministic
    and persistent**:

    1. If a seed already lives in the OS Keychain, load and return it (the app
       loads the *same* seed on every boot).
    2. Otherwise generate one, persist it to the Keychain, and echo the
       ``otpauth://`` provisioning URI to the console **once** for the CEO to
       scan. Subsequent boots take path (1).

    ``echo`` is injectable (defaults to ``print``) so tests can capture the
    provisioning output without writing to stdout.
    """
    try:
        return load_secret(CEO_TOTP_SECRET)
    except SecretNotFoundError:
        secret = generate_totp_secret()
        store_ceo_totp_secret(secret)
        echo(
            "[AFML] No CEO TOTP seed found — generated a new one and stored it in "
            f"the Keychain (service={KEYCHAIN_SERVICE!r}). Scan this ONCE into your "
            "authenticator app; it persists across restarts:"
        )
        echo(provisioning_uri(secret, account_name=account_name, issuer=issuer))
        return secret
