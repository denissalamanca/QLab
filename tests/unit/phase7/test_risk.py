"""Phase 7 — risk engine (concurrent scaling + ESMA caps + FTMO buffer)."""

from __future__ import annotations

import pytest

from afml.config.assets import AssetClass
from afml.config.risk import (
    ESMA_MAX_LEVERAGE,
    FTMO_MAX_DRAWDOWN_BUFFER,
    margin_fraction_for,
)
from afml.execution.risk import RiskEngine


@pytest.mark.phase7
def test_esma_leverage_tiers() -> None:
    """FX 30:1, index/metal 20:1, crypto 2:1 (ESMA retail caps)."""
    assert ESMA_MAX_LEVERAGE[AssetClass.FX] == 30.0
    assert ESMA_MAX_LEVERAGE[AssetClass.INDEX] == 20.0
    assert ESMA_MAX_LEVERAGE[AssetClass.METAL] == 20.0
    assert ESMA_MAX_LEVERAGE[AssetClass.CRYPTO] == 2.0


@pytest.mark.phase7
def test_margin_fraction_is_inverse_leverage() -> None:
    assert margin_fraction_for(AssetClass.FX) == pytest.approx(1.0 / 30.0)
    assert margin_fraction_for(AssetClass.CRYPTO) == pytest.approx(0.5)


@pytest.mark.phase7
def test_concurrent_scaling_divides_by_c95() -> None:
    """The scaled size is ``raw / c95``."""
    engine = RiskEngine(account_equity=100_000.0, c95=10.0)
    bet = engine.size_bet(1.0, AssetClass.FX, commit=False)
    assert bet.scaled_size == pytest.approx(0.1)


@pytest.mark.phase7
def test_margin_uses_esma_fraction() -> None:
    """A full-confidence FX bet with c95=1 commits ``1/30`` of equity."""
    equity = 100_000.0
    engine = RiskEngine(account_equity=equity, c95=1.0)
    bet = engine.size_bet(1.0, AssetClass.FX, commit=False)
    # Without the buffer clamp this would be equity/30 ≈ 3333; but the buffer
    # is 10% = 10000, so 3333 < 10000 → not clamped.
    assert bet.margin == pytest.approx(equity / 30.0)
    assert not bet.clamped


@pytest.mark.phase7
def test_crypto_commits_more_margin_than_fx() -> None:
    """Crypto's 2:1 cap demands far more margin than FX's 30:1 for the same
    bet size."""
    equity = 100_000.0
    fx_engine = RiskEngine(account_equity=equity, c95=1.0)
    crypto_engine = RiskEngine(account_equity=equity, c95=1.0)
    fx = fx_engine.size_bet(0.1, AssetClass.FX, commit=False)
    crypto = crypto_engine.size_bet(0.1, AssetClass.CRYPTO, commit=False)
    assert crypto.margin > fx.margin


@pytest.mark.phase7
def test_running_margin_never_exceeds_buffer() -> None:
    """Repeated max-size bets must clamp at the FTMO buffer."""
    equity = 100_000.0
    buffer = FTMO_MAX_DRAWDOWN_BUFFER * equity
    engine = RiskEngine(account_equity=equity, c95=1.0)
    for _ in range(100):
        engine.size_bet(1.0, AssetClass.FX)
    assert engine.committed_margin <= buffer + 1e-6


@pytest.mark.phase7
def test_budget_exhaustion_yields_zero_margin_bets() -> None:
    """Once the buffer is consumed, further bets size to 0 margin."""
    equity = 100_000.0
    engine = RiskEngine(account_equity=equity, c95=1.0)
    # Consume the whole budget with crypto bets (50% margin each).
    engine.size_bet(1.0, AssetClass.CRYPTO)  # commits 50% → but buffer is 10%
    assert engine.remaining_budget == pytest.approx(0.0)
    later = engine.size_bet(1.0, AssetClass.FX)
    assert later.margin == 0.0
    assert later.clamped


@pytest.mark.phase7
def test_reset_clears_committed_margin() -> None:
    engine = RiskEngine(account_equity=100_000.0, c95=2.0)
    engine.size_bet(1.0, AssetClass.FX)
    assert engine.committed_margin > 0.0
    engine.reset()
    assert engine.committed_margin == 0.0


@pytest.mark.phase7
def test_risk_engine_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="account_equity"):
        RiskEngine(account_equity=0.0)
    with pytest.raises(ValueError, match="c95"):
        RiskEngine(account_equity=100.0, c95=0.5)
    with pytest.raises(ValueError, match="drawdown_buffer_pct"):
        RiskEngine(account_equity=100.0, drawdown_buffer_pct=1.5)
