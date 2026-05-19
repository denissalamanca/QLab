"""Settings for the production control-plane entrypoint.

Read from the environment (prefix ``AFML_CP_``). The CEO **public** key is
non-secret config; the TOTP secret is loaded from the Keychain at wiring time
(see ``apps/api/main.py``), never from the environment.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ControlPlaneSettings(BaseSettings):
    """Environment-driven configuration for the control-plane server."""

    model_config = SettingsConfigDict(env_prefix="AFML_CP_", extra="ignore")

    #: Raw Ed25519 public key, hex-encoded (64 hex chars / 32 bytes).
    ceo_public_key_hex: str = ""
    #: SQLAlchemy URL for the Alpha Registry.
    registry_db_url: str = "sqlite:///afml_registry.db"
    #: Account equity seeded into the risk engine for the flatten path.
    account_equity: float = 100_000.0
    #: Historical 95th-percentile concurrent-trade count.
    c95: float = 1.0
    #: Allowed browser origins for the React frontend (comma-separated).
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
