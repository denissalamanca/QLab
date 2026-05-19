"""TOTP second factor for CEO authorisation (§11.2 mandatory 2FA).

A valid Ed25519 signature proves *possession of the private key*; the TOTP
code proves *live presence* of the CEO at authorisation time. Both are
required for any capital-allocating action — a stolen key alone cannot move
capital without the rotating 6-digit code from the CEO's authenticator app.

Thin wrapper over ``pyotp`` with the AFML-canonical settings (SHA-1, 6
digits, 30-second step — the de-facto authenticator-app standard) baked in.
"""

from __future__ import annotations

import pyotp

# RFC 6238 defaults — match Google Authenticator / 1Password / Authy.
TOTP_DIGITS: int = 6
TOTP_INTERVAL_SECONDS: int = 30
# Accept the adjacent window each side to tolerate clock skew / entry latency.
DEFAULT_VALID_WINDOW: int = 1


def generate_totp_secret() -> str:
    """Generate a fresh base32 TOTP secret for provisioning an authenticator."""
    return pyotp.random_base32()


def provisioning_uri(secret: str, *, account_name: str, issuer: str = "AFML Quant Lab") -> str:
    """Build the ``otpauth://`` URI to render as a QR code during enrolment."""
    return pyotp.TOTP(secret, digits=TOTP_DIGITS, interval=TOTP_INTERVAL_SECONDS).provisioning_uri(
        name=account_name, issuer_name=issuer
    )


def current_totp(secret: str) -> str:
    """Return the TOTP code valid *right now* (used in tests / a CLI helper)."""
    return pyotp.TOTP(secret, digits=TOTP_DIGITS, interval=TOTP_INTERVAL_SECONDS).now()


def verify_totp(secret: str, code: str, *, valid_window: int = DEFAULT_VALID_WINDOW) -> bool:
    """Verify a submitted TOTP ``code`` against the ``secret``.

    Parameters
    ----------
    secret
        Base32 TOTP secret.
    code
        The 6-digit code the CEO entered.
    valid_window
        Number of adjacent 30-second steps to accept on each side (clock
        skew tolerance). ``1`` ⇒ accepts the previous, current, and next code.

    Returns
    -------
    ``True`` iff the code matches within the window. Empty / malformed codes
    return ``False`` rather than raising.
    """
    if not code or not code.strip():
        return False
    totp = pyotp.TOTP(secret, digits=TOTP_DIGITS, interval=TOTP_INTERVAL_SECONDS)
    return bool(totp.verify(code.strip(), valid_window=valid_window))
