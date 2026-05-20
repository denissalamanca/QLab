"""Phase 9 — AFML 0-9 final polishing audit edge cases (P1-P4).

- **P1** GMM has no ``.cdf()``; the bet sizer computes the mixture CDF manually
  as a weighted sum of component normal CDFs (validated against an independent
  recompute).
- **P2** GSADF runs off the event loop via ``run_in_executor`` and returns the
  same result as the synchronous path.
- **P3** the CEO TOTP seed is persisted (Keychain) and survives a "reboot".
- **P4** CORS preflight from the Vite origin is allowed (dashboard not blocked).
"""

from __future__ import annotations

import asyncio

import keyring
import numpy as np
import numpy.typing as npt
import pytest
from fastapi.testclient import TestClient
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from afml.control_plane import create_app
from afml.crypto import (
    CEO_TOTP_SECRET,
    current_totp,
    get_or_create_ceo_totp_secret,
    verify_totp,
)
from afml.execution.bet_sizing import _mixture_cdf_sizes, bet_sizes_for_batch
from afml.monitoring.pipeline import StructuralBreakMonitor

pytestmark = pytest.mark.phase9


# ----------------------------------------------------------- P1: GMM CDF trap
def test_gaussian_mixture_has_no_cdf_method() -> None:
    """Documents the trap: sklearn's GaussianMixture exposes no .cdf()."""
    gm = GaussianMixture(n_components=2)
    assert not hasattr(gm, "cdf")


def _fit_bimodal_gmm(seed: int = 0) -> tuple[GaussianMixture, npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    z = np.concatenate([rng.normal(-2.0, 0.3, 60), rng.normal(2.0, 0.3, 60)]).astype(np.float64)
    gm = GaussianMixture(n_components=2, random_state=seed).fit(z.reshape(-1, 1))
    return gm, z


def test_mixture_cdf_matches_manual_weighted_sum() -> None:
    """``_mixture_cdf_sizes`` == 2·Σ_k w_k·Φ((z-μ_k)/σ_k) − 1, recomputed independently."""
    gm, _ = _fit_bimodal_gmm()
    z_eval = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float64)

    got = _mixture_cdf_sizes(z_eval, gm)

    weights = gm.weights_.ravel()
    means = gm.means_.ravel()
    stds = np.sqrt(gm.covariances_.ravel())
    manual = np.array([
        2.0
        * sum(w * norm.cdf(z, loc=m, scale=s) for w, m, s in zip(weights, means, stds, strict=True))
        - 1.0
        for z in z_eval
    ])
    assert np.allclose(got, manual, atol=1e-12)


def test_mixture_cdf_is_monotone_and_bounded() -> None:
    gm, _ = _fit_bimodal_gmm()
    z_eval = np.linspace(-5.0, 5.0, 50, dtype=np.float64)
    sizes = _mixture_cdf_sizes(z_eval, gm)
    assert np.all(sizes >= -1.0 - 1e-9) and np.all(sizes <= 1.0 + 1e-9)
    assert np.all(np.diff(sizes) >= -1e-9)  # non-decreasing in z


def test_bimodal_batch_triggers_mixture_without_attribute_error() -> None:
    """The end-to-end batch sizer engages the mixture CDF and stays in [0, 1]
    (i.e. never calls the non-existent gmm.cdf)."""
    rng = np.random.default_rng(1)
    probs = np.clip(
        np.concatenate([rng.normal(0.58, 0.01, 30), rng.normal(0.92, 0.01, 30)]), 0.0, 1.0
    )
    out = bet_sizes_for_batch(probs)
    assert out.used_mixture_fallback
    assert np.all(out.sizes >= 0.0) and np.all(out.sizes <= 1.0)


# --------------------------------------------------------- P2: GSADF offload
async def test_check_regime_async_matches_sync() -> None:
    """The executor-offloaded GSADF returns the identical result as inline."""
    rng = np.random.default_rng(7)
    prices = (np.cumsum(rng.normal(0.0, 1.0, 90)) + 100.0).astype(np.float64)
    monitor = StructuralBreakMonitor(n_simulations=25)

    sync = monitor.check_regime("EURUSD", prices, random_state=3)
    asyncv = await monitor.check_regime_async("EURUSD", prices, random_state=3)

    assert asyncv.asset == sync.asset
    assert asyncv.gsadf.gsadf_statistic == pytest.approx(sync.gsadf.gsadf_statistic)
    assert asyncv.regime_break == sync.regime_break


async def test_check_regime_async_runs_concurrently() -> None:
    """Several offloaded GSADF calls complete concurrently without blocking."""
    rng = np.random.default_rng(11)
    monitor = StructuralBreakMonitor(n_simulations=20)
    series = [(np.cumsum(rng.normal(0.0, 1.0, 80)) + 100.0).astype(np.float64) for _ in range(4)]
    results = await asyncio.gather(*[
        monitor.check_regime_async(f"A{i}", s, random_state=i) for i, s in enumerate(series)
    ])
    assert [r.asset for r in results] == ["A0", "A1", "A2", "A3"]


# ----------------------------------------------------- P3: persistent TOTP seed
@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    store: dict[tuple[str, str], str] = {}

    def set_password(service: str, name: str, value: str) -> None:
        store[(service, name)] = value

    def get_password(service: str, name: str) -> str | None:
        return store.get((service, name))

    monkeypatch.setattr(keyring, "set_password", set_password)
    monkeypatch.setattr(keyring, "get_password", get_password)
    return store


def test_totp_seed_persists_across_reboots(fake_keyring: dict[tuple[str, str], str]) -> None:
    captured: list[str] = []

    # First boot: no seed yet → generate, persist, display the provisioning URI once.
    secret1 = get_or_create_ceo_totp_secret(echo=captured.append)
    assert any("otpauth://" in line for line in captured)
    assert ("afml-quant-lab", CEO_TOTP_SECRET) in fake_keyring

    # Second boot: the SAME seed is loaded, nothing new echoed.
    captured.clear()
    secret2 = get_or_create_ceo_totp_secret(echo=captured.append)
    assert secret2 == secret1
    assert captured == []

    # The persisted seed is a usable TOTP secret.
    assert verify_totp(secret2, current_totp(secret2))


# ------------------------------------------------------------- P4: CORS preflight
def test_cors_preflight_allows_vite_origin() -> None:
    client = TestClient(create_app())
    resp = client.options(
        "/api/v1/registry/strategies",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert resp.headers["access-control-allow-credentials"] == "true"


def test_cors_get_carries_allow_origin_header() -> None:
    client = TestClient(create_app())
    resp = client.get("/api/v1/health", headers={"Origin": "http://127.0.0.1:5173"})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_cors_rejects_unknown_origin() -> None:
    client = TestClient(create_app())
    resp = client.get("/api/v1/health", headers={"Origin": "http://evil.example.com"})
    # Endpoint still answers, but the browser-enforced allow-origin is NOT granted.
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers
