"""Operational health checks — the engine behind ``afml doctor`` (Ops M0).

Each check is a small, independently-testable function returning a :class:`Check`.
``run_health_checks`` wires the production checks (real asset registry + settings
+ Keychain + Redis); tests exercise the primitives with temp files / fakes.

The warm-up check is a **coarse availability gate** (is the tick file plausibly
complete?). The *precise* warm-up alignment — that every Phase-2 event clears the
FFD truncation + the largest feature window — is enforced by the M1 harness.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from sqlalchemy import text

from afml.config.assets import ASSETS, missing_oos_files
from afml.config.settings import Settings, get_settings
from afml.core.broker import MessageBroker
from afml.core.registry.repository import AlphaRegistryRepository
from afml.crypto import (
    CEO_PRIVATE_KEY,
    CEO_TOTP_SECRET,
    SecretNotFoundError,
    load_secret,
)
from afml.features import DEFAULT_WINDOWS

# Coarse availability floor. A complete tick file must comfortably exceed the
# bars needed to clear the largest feature window + the Phase-2 minimum event
# count; we require ≥ ~50 ticks per such bar of raw history. Real Dukascopy
# files hold millions — this only catches a truncated / empty download.
_MIN_USABLE_BARS: int = max(DEFAULT_WINDOWS) + 500
DEFAULT_MIN_TICKS: int = 50 * _MIN_USABLE_BARS


@dataclass(frozen=True, slots=True)
class Check:
    """One health-check result."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Aggregate of all checks."""

    checks: tuple[Check, ...]

    @property
    def healthy(self) -> bool:
        return all(c.ok for c in self.checks)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]


def check_files_present(name: str, paths: Iterable[Path]) -> Check:
    """All ``paths`` must exist on disk."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        head = missing[:3]
        suffix = " …" if len(missing) > 3 else ""
        return Check(name, False, f"{len(missing)} missing: {head}{suffix}")
    return Check(name, True, "all present")


def check_warmup(name: str, paths: Iterable[Path], *, min_ticks: int = DEFAULT_MIN_TICKS) -> Check:
    """Each existing parquet must hold ≥ ``min_ticks`` rows (coarse warm-up gate)."""
    thin: list[str] = []
    unreadable: list[str] = []
    for p in paths:
        if not p.exists():
            unreadable.append(f"{p.name} (absent)")
            continue
        try:
            n = int(pl.scan_parquet(p).select(pl.len()).collect().item())
        except Exception as exc:
            unreadable.append(f"{p.name} ({exc})")
            continue
        if n < min_ticks:
            thin.append(f"{p.name} ({n} < {min_ticks})")
    problems = unreadable + thin
    if problems:
        return Check(
            name, False, f"insufficient: {problems[:3]}{' …' if len(problems) > 3 else ''}"
        )
    return Check(name, True, f"≥ {min_ticks} ticks each")


def check_keychain(name: str, secret_names: Iterable[str]) -> Check:
    """Every named secret must be present in the OS Keychain."""
    missing = []
    for s in secret_names:
        try:
            load_secret(s)
        except SecretNotFoundError:
            missing.append(s)
    if missing:
        return Check(name, False, f"missing secrets: {missing} (run `afml enroll-ceo`)")
    return Check(name, True, "present")


def check_redis(name: str = "redis bus") -> Check:
    """The message bus must accept a connection + PING."""

    async def _ping() -> None:
        broker = await MessageBroker.connect()
        await broker.close()

    try:
        asyncio.run(_ping())
    except Exception as exc:
        return Check(name, False, f"unreachable: {exc}")
    return Check(name, True, "reachable")


def check_registry(name: str, db_url: str) -> Check:
    """The Alpha Registry must be reachable (a trivial SELECT succeeds)."""
    try:
        repo = AlphaRegistryRepository(db_url)
        with repo.session() as s:
            s.execute(text("SELECT 1"))
    except Exception as exc:
        return Check(name, False, f"unreachable: {exc}")
    return Check(name, True, "reachable")


def run_health_checks(
    *,
    settings: Settings | None = None,
    min_ticks: int = DEFAULT_MIN_TICKS,
) -> HealthReport:
    """Production wiring: research + OOS data, warm-up, Redis, registry, Keychain."""
    cfg = settings or get_settings()
    research = [a.data_path for a in ASSETS]
    oos_missing = missing_oos_files()
    oos_check = (
        Check("OOS data (2026 YTD)", True, "all present")
        if not oos_missing
        else Check("OOS data (2026 YTD)", False, f"missing for: {oos_missing}")
    )
    return HealthReport((
        check_files_present("research data (2020-2025)", research),
        oos_check,
        check_warmup("warm-up history", research, min_ticks=min_ticks),
        check_redis(),
        check_registry("alpha registry", cfg.registry_db_url),
        check_keychain("CEO Keychain secrets", [CEO_PRIVATE_KEY, CEO_TOTP_SECRET]),
    ))
