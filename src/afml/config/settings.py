"""Centralized runtime configuration via Pydantic Settings.

Every value that may differ across dev/test/prod lives here and is sourced from
environment variables (prefix ``AFML_``) or a local ``.env`` file. This is the
only legal home for runtime constants — magic literals scattered in the codebase
violate the AFML anti-bias rule.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AFML_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths -----------------------------------------------------------------
    data_root: Path = Field(
        default=Path("/Users/dsalamanca/vs_env/Antigravity/Quant Lab/data/multi_year_consolidated"),
        description="Root directory of raw Dukascopy parquet files.",
    )
    artifact_root: Path = Field(
        default=Path("./artifacts"),
        description="Working directory for derived datasets, models, registries.",
    )

    # --- Alpha Registry --------------------------------------------------------
    registry_db_url: str = Field(
        default="sqlite:///./artifacts/alpha_registry.db",
        description="SQLAlchemy URL for the Historical Alpha Registry.",
    )
    registry_wal_mode: bool = Field(default=True)

    # --- Redis (message broker) ------------------------------------------------
    redis_url: str = Field(default="redis://localhost:6379/0")

    # --- Operating mode --------------------------------------------------------
    use_fake_redis: bool = Field(
        default=False,
        description="If true, use in-memory fakeredis instead of real Redis. "
        "Tests set this to true automatically.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton accessor."""
    return Settings()
