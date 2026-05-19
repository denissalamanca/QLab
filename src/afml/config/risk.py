"""Regulatory & prop-firm risk constants (Blueprint §9.2).

ESMA retail leverage caps and FTMO prop-account drawdown rules are *domain
constants* — they belong here, never inline in the execution code (AFML
anti-bias rule: domain constants live in ``src/afml/config/``).

**ESMA retail leverage** (ESMA/2018/796 product-intervention limits, expressed
as the maximum leverage and the equivalent minimum margin fraction):

| Asset class            | Max leverage | Margin fraction |
|------------------------|-------------:|----------------:|
| Major FX               | 30:1         | 3.33 %          |
| Non-major FX / gold / major index | 20:1 | 5 %        |
| Other commodities / non-major index | 10:1 | 10 %     |
| Crypto                 | 2:1          | 50 %            |

We map our 4-class :class:`afml.config.assets.AssetClass` onto these tiers
conservatively: all FX → 30:1, indices → 20:1, metals (gold/silver) → 20:1,
crypto → 2:1.

**FTMO** challenge / funded rules: max **5 %** daily loss, max **10 %** total
drawdown. The execution layer treats the 10 % total drawdown as the hard
exposure ceiling — the sum of margin committed across all concurrent positions
must never exceed this buffer.
"""

from __future__ import annotations

from afml.config.assets import AssetClass

# --- ESMA maximum leverage by asset class ---------------------------------------
ESMA_MAX_LEVERAGE: dict[AssetClass, float] = {
    AssetClass.FX: 30.0,
    AssetClass.INDEX: 20.0,
    AssetClass.METAL: 20.0,
    AssetClass.CRYPTO: 2.0,
}

# Equivalent minimum margin fraction (= 1 / leverage).
ESMA_MARGIN_FRACTION: dict[AssetClass, float] = {
    cls: 1.0 / lev for cls, lev in ESMA_MAX_LEVERAGE.items()
}

# --- FTMO prop-account limits ---------------------------------------------------
FTMO_MAX_DAILY_LOSS_PCT: float = 0.05
FTMO_MAX_TOTAL_DRAWDOWN_PCT: float = 0.10
# The execution engine's hard ceiling on total committed margin, as a fraction
# of account equity. Equal to the FTMO total-drawdown buffer — committing more
# margin than the worst tolerable loss would risk an instant rule breach.
FTMO_MAX_DRAWDOWN_BUFFER: float = FTMO_MAX_TOTAL_DRAWDOWN_PCT


def max_leverage_for(asset_class: AssetClass) -> float:
    """ESMA maximum leverage for an asset class."""
    return ESMA_MAX_LEVERAGE[asset_class]


def margin_fraction_for(asset_class: AssetClass) -> float:
    """Minimum margin fraction (``1 / leverage``) for an asset class."""
    return ESMA_MARGIN_FRACTION[asset_class]
