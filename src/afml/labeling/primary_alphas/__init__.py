"""Brain 1 primary-alpha plugin family.

Three plugins are mandatory (per locked decision — orthogonality diversity):
- ``SymmetricCUSUM``         — structural volatility-event detector
- ``BollingerMeanReversion`` — ranging-market mean reversion
- ``DonchianBreakout``       — momentum breakout

Agent 2 sweeps all three families and logs every hypothesis tested to the
Alpha Registry. The registry's orthogonality check rejects any new family that
duplicates an already-deployed strategy.
"""

from afml.labeling.primary_alphas.base import (
    PrimaryAlpha,
    PrimaryAlphaParams,
    get_alpha_class,
    list_alpha_families,
    register_alpha,
)
from afml.labeling.primary_alphas.bollinger import BollingerMeanReversion
from afml.labeling.primary_alphas.cusum import SymmetricCUSUM
from afml.labeling.primary_alphas.donchian import DonchianBreakout

__all__ = [
    "BollingerMeanReversion",
    "DonchianBreakout",
    "PrimaryAlpha",
    "PrimaryAlphaParams",
    "SymmetricCUSUM",
    "get_alpha_class",
    "list_alpha_families",
    "register_alpha",
]
