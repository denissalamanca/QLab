"""Phase 9 — Keychain façade (Blueprint §11).

Uses a dict-backed fake ``keyring`` so the suite never touches the real OS
Keychain (which would prompt or fail in CI).
"""

from __future__ import annotations

import keyring
import pytest

from afml.crypto import (
    CEO_PRIVATE_KEY,
    CEO_TOTP_SECRET,
    SecretNotFoundError,
    delete_secret,
    load_secret,
    store_ceo_private_key,
    store_ceo_totp_secret,
    store_secret,
)

pytestmark = pytest.mark.phase9


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    store: dict[tuple[str, str], str] = {}

    def set_password(service: str, name: str, value: str) -> None:
        store[(service, name)] = value

    def get_password(service: str, name: str) -> str | None:
        return store.get((service, name))

    def delete_password(service: str, name: str) -> None:
        if (service, name) in store:
            del store[(service, name)]
        else:
            raise keyring.errors.PasswordDeleteError("not found")

    monkeypatch.setattr(keyring, "set_password", set_password)
    monkeypatch.setattr(keyring, "get_password", get_password)
    monkeypatch.setattr(keyring, "delete_password", delete_password)
    return store


def test_store_and_load_roundtrip(fake_keyring: dict[tuple[str, str], str]) -> None:
    store_secret("my_key", "my_value")
    assert load_secret("my_key") == "my_value"


def test_load_missing_raises(fake_keyring: dict[tuple[str, str], str]) -> None:
    with pytest.raises(SecretNotFoundError):
        load_secret("does_not_exist")


def test_delete_is_idempotent(fake_keyring: dict[tuple[str, str], str]) -> None:
    store_secret("k", "v")
    delete_secret("k")
    delete_secret("k")  # second delete must not raise
    with pytest.raises(SecretNotFoundError):
        load_secret("k")


def test_ceo_secret_helpers(fake_keyring: dict[tuple[str, str], str]) -> None:
    store_ceo_private_key("deadbeef")
    store_ceo_totp_secret("JBSWY3DPEHPK3PXP")
    assert load_secret(CEO_PRIVATE_KEY) == "deadbeef"
    assert load_secret(CEO_TOTP_SECRET) == "JBSWY3DPEHPK3PXP"
