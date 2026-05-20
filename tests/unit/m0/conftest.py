"""M0 fixtures — a dict-backed fake Keychain so enrol/doctor never touch the OS."""

from __future__ import annotations

import keyring
import pytest


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
