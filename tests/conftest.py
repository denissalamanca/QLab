"""Top-level pytest configuration.

All tests default to in-memory ``fakeredis``; tests needing a real broker (the
10K msg/sec throughput benchmark) live under ``tests/benchmarks/`` and clear the
override explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from afml.config import settings as _settings_module


@pytest.fixture(autouse=True)
def _force_fake_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``use_fake_redis=True`` for every test unless explicitly overridden."""
    monkeypatch.setenv("AFML_USE_FAKE_REDIS", "true")
    # Invalidate the lru_cache on get_settings so the new env var is picked up.
    _settings_module.get_settings.cache_clear()


@pytest.fixture
def tmp_db_url(tmp_path: Path) -> str:
    """SQLite URL pointing at a unique per-test temporary database."""
    return f"sqlite:///{tmp_path / 'test_registry.db'}"
