"""CEO authorisation core for the Phase 9 control plane (Blueprint §11.2).

Every capital-allocating control-plane action is gated here. The
:class:`CEOAuthenticator` holds only the material the *server* legitimately
needs to verify a request:

- the CEO's Ed25519 **public** key (verification only — the private key never
  leaves the CEO's signing device / Keychain), and
- the shared **TOTP secret** (server-side, to validate the rotating code).

Two distinct trust levels, matching the Phase 0 event contracts:

- **Approval** (``CEOApproval``) promotes a strategy Paper → Live — it *commits
  capital*, so it requires BOTH a valid signature over the canonical approval
  message AND a live TOTP code (§11.2 mandatory 2FA).
- **Flatten** (``EmergencyFlatten``) is a risk-*reducing* kill-switch — it
  requires a valid signature (proves possession of the CEO key) but NOT a TOTP
  code, so the emergency path is never blocked by a 30-second code window. A
  per-nonce replay guard stops a captured flatten signature from being
  re-fired.

Verification failures raise :class:`CEOAuthError` subclasses; the API layer
maps them to HTTP 403.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from afml.crypto.signing import approval_message, flatten_message, verify_signature
from afml.crypto.totp import DEFAULT_VALID_WINDOW, verify_totp

# AFML 0-9 final audit V1: signatures are valid only within ±60 s of the
# signed timestamp, bounding any replay window (a network retry or captured
# payload) to one minute even before the per-nonce guard.
DEFAULT_MAX_TIMESTAMP_SKEW_MS: int = 60_000


def _now_ms() -> int:
    """Current UTC time in milliseconds (the server clock for freshness checks)."""
    return int(time.time() * 1000)


class CEOAuthError(Exception):
    """Base class for any CEO-authorisation failure → HTTP 403."""


class InvalidSignatureError(CEOAuthError):
    """The Ed25519 signature is missing, malformed, or does not verify."""


class InvalidTOTPError(CEOAuthError):
    """The TOTP second factor is missing or does not match the window."""


class ReplayError(CEOAuthError):
    """A flatten nonce was reused — possible replay of a captured signature."""


class StaleTimestampError(CEOAuthError):
    """The signed timestamp is outside the freshness window — possible replay."""


@dataclass
class CEOAuthenticator:
    """Server-side verifier for CEO-signed control-plane actions.

    Parameters
    ----------
    public_key_bytes
        Raw 32-byte Ed25519 public key. Public material — safe in config.
    totp_secret
        Base32 TOTP shared secret (server-side; load from the Keychain in
        production via :func:`afml.crypto.load_secret`).
    totp_valid_window
        Adjacent 30-second steps accepted on each side (clock-skew tolerance).
    """

    public_key_bytes: bytes
    totp_secret: str
    totp_valid_window: int = DEFAULT_VALID_WINDOW
    max_timestamp_skew_ms: int = DEFAULT_MAX_TIMESTAMP_SKEW_MS
    time_provider: Callable[[], int] = _now_ms
    _consumed_nonces: set[str] = field(default_factory=set, repr=False)

    def _check_timestamp(self, timestamp_ms: int) -> None:
        """Reject a signed timestamp outside the ±``max_timestamp_skew_ms`` window."""
        skew = abs(self.time_provider() - timestamp_ms)
        if skew > self.max_timestamp_skew_ms:
            raise StaleTimestampError(
                f"timestamp skew {skew} ms exceeds {self.max_timestamp_skew_ms} ms window"
            )

    def verify_approval(
        self,
        experiment_id: UUID,
        timestamp_ms: int,
        signed_token: str,
        totp_code: str,
    ) -> None:
        """Authorise a Paper → Live promotion. Requires signature **and** TOTP.

        The signature must be over ``afml:approve:<experiment_id>:<timestamp_ms>``
        and the timestamp must be fresh (±60 s) — see audit V1.

        Raises
        ------
        StaleTimestampError
            If ``timestamp_ms`` is outside the freshness window (replay guard).
        InvalidSignatureError
            If ``signed_token`` does not verify against the canonical message.
        InvalidTOTPError
            If ``totp_code`` is empty / malformed / outside the window.
        """
        self._check_timestamp(timestamp_ms)
        message = approval_message(str(experiment_id), timestamp_ms)
        if not verify_signature(self.public_key_bytes, message, signed_token):
            raise InvalidSignatureError("invalid or missing Ed25519 signature")
        if not verify_totp(self.totp_secret, totp_code, valid_window=self.totp_valid_window):
            raise InvalidTOTPError("invalid or missing TOTP code")

    def verify_flatten(self, nonce: str, timestamp_ms: int, signed_token: str) -> None:
        """Authorise an emergency flatten. Requires signature only.

        Defence in depth (audit V1): the signature is over
        ``afml:flatten:<nonce>:<timestamp_ms>``. The ``timestamp_ms`` must be
        fresh (±60 s) **and** the ``nonce`` single-use — so a captured flatten
        signature is replayable neither within the window (nonce consumed) nor
        after it (timestamp stale), even across a restart that clears the nonce
        set. A nonce is consumed only after both checks pass.

        Raises
        ------
        StaleTimestampError
            If ``timestamp_ms`` is outside the freshness window.
        InvalidSignatureError
            If ``signed_token`` does not verify against the canonical message.
        ReplayError
            If ``nonce`` has already been used.
        """
        if not nonce or not nonce.strip():
            raise InvalidSignatureError("flatten nonce must be a non-empty string")
        self._check_timestamp(timestamp_ms)
        message = flatten_message(nonce, timestamp_ms)
        if not verify_signature(self.public_key_bytes, message, signed_token):
            raise InvalidSignatureError("invalid or missing Ed25519 signature")
        if nonce in self._consumed_nonces:
            raise ReplayError("flatten nonce already used — replay rejected")
        self._consumed_nonces.add(nonce)
