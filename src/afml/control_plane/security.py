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

from dataclasses import dataclass, field
from uuid import UUID

from afml.crypto.signing import approval_message, flatten_message, verify_signature
from afml.crypto.totp import DEFAULT_VALID_WINDOW, verify_totp


class CEOAuthError(Exception):
    """Base class for any CEO-authorisation failure → HTTP 403."""


class InvalidSignatureError(CEOAuthError):
    """The Ed25519 signature is missing, malformed, or does not verify."""


class InvalidTOTPError(CEOAuthError):
    """The TOTP second factor is missing or does not match the window."""


class ReplayError(CEOAuthError):
    """A flatten nonce was reused — possible replay of a captured signature."""


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
    _consumed_nonces: set[str] = field(default_factory=set, repr=False)

    def verify_approval(self, experiment_id: UUID, signed_token: str, totp_code: str) -> None:
        """Authorise a Paper → Live promotion. Requires signature **and** TOTP.

        Raises
        ------
        InvalidSignatureError
            If ``signed_token`` does not verify against the canonical
            ``afml:approve:<experiment_id>`` message.
        InvalidTOTPError
            If ``totp_code`` is empty / malformed / outside the window.
        """
        message = approval_message(str(experiment_id))
        if not verify_signature(self.public_key_bytes, message, signed_token):
            raise InvalidSignatureError("invalid or missing Ed25519 signature")
        if not verify_totp(self.totp_secret, totp_code, valid_window=self.totp_valid_window):
            raise InvalidTOTPError("invalid or missing TOTP code")

    def verify_flatten(self, nonce: str, signed_token: str) -> None:
        """Authorise an emergency flatten. Requires signature only.

        The ``nonce`` binds the signature to a single use; reusing it raises
        :class:`ReplayError`. A nonce is consumed only after the signature
        verifies, so a bad-signature attempt cannot burn a legitimate nonce.

        Raises
        ------
        InvalidSignatureError
            If ``signed_token`` does not verify against ``afml:flatten:<nonce>``.
        ReplayError
            If ``nonce`` has already been used.
        """
        if not nonce or not nonce.strip():
            raise InvalidSignatureError("flatten nonce must be a non-empty string")
        message = flatten_message(nonce)
        if not verify_signature(self.public_key_bytes, message, signed_token):
            raise InvalidSignatureError("invalid or missing Ed25519 signature")
        if nonce in self._consumed_nonces:
            raise ReplayError("flatten nonce already used — replay rejected")
        self._consumed_nonces.add(nonce)
