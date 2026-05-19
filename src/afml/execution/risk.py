"""Risk sizing — concurrent scaling + ESMA leverage + FTMO margin cap (§9.2).

Three layers convert a raw bet size ``∈ [0, 1]`` (from
:mod:`afml.execution.bet_sizing`) into a deployable margin commitment:

1. **Concurrent-position scaling.** Divide the raw size by ``c_95`` — the
   historical 95th-percentile count of simultaneously-active trades — so that
   a *typical* burst of concurrent signals sums to ≤ one unit of exposure
   (Blueprint §9.2).

2. **ESMA leverage cap.** Translate the scaled fraction into margin using the
   asset class's ESMA margin fraction (``1 / leverage``). A size-1.0 bet on a
   30:1 FX pair commits ``3.33 %`` of the allocated capital as margin.

3. **FTMO drawdown buffer (hard cap).** The ``c_95`` heuristic protects the
   *typical* case; it does NOT bound a pathological burst (e.g. 50 signals
   firing at once when ``c_95 = 10``). So the engine also enforces an absolute
   ceiling: the running sum of committed margin may never exceed
   ``FTMO_MAX_DRAWDOWN_BUFFER × equity``. New bets are clamped against the
   remaining budget; once the budget is exhausted, further bets size to 0.
   This is what guarantees the Blueprint §9.3 "50 concurrent max-confidence
   signals" test stays within the buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

from afml.config.assets import AssetClass
from afml.config.risk import (
    FTMO_MAX_DRAWDOWN_BUFFER,
    margin_fraction_for,
)

DEFAULT_C95: float = 1.0


@dataclass(frozen=True, slots=True)
class SizedBet:
    """A bet after concurrent scaling + leverage + buffer clamping.

    Attributes
    ----------
    raw_size
        The input bet size in ``[0, 1]`` (pre-risk).
    scaled_size
        Size after ``c_95`` concurrent-position scaling.
    margin
        Margin (in account-currency units) the bet commits, after the ESMA
        leverage translation AND the FTMO buffer clamp.
    clamped
        True iff the FTMO buffer clamp reduced the margin below what the
        scaled size + leverage would otherwise demand.
    """

    raw_size: float
    scaled_size: float
    margin: float
    clamped: bool


@dataclass(slots=True)
class RiskEngine:
    """Stateful per-account risk sizer.

    Tracks cumulative committed margin across a burst of bets and enforces the
    FTMO drawdown buffer as a hard ceiling.

    Parameters
    ----------
    account_equity
        Total account equity in account currency.
    c95
        Historical 95th-percentile concurrent-trade count. Must be ≥ 1.
    drawdown_buffer_pct
        Fraction of equity usable as total committed margin (default
        :data:`FTMO_MAX_DRAWDOWN_BUFFER` = 0.10).
    """

    account_equity: float
    c95: float = DEFAULT_C95
    drawdown_buffer_pct: float = FTMO_MAX_DRAWDOWN_BUFFER
    _committed_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.account_equity <= 0:
            raise ValueError(f"account_equity must be > 0, got {self.account_equity}")
        if self.c95 < 1.0:
            raise ValueError(f"c95 must be ≥ 1, got {self.c95}")
        if not 0.0 < self.drawdown_buffer_pct <= 1.0:
            raise ValueError(
                f"drawdown_buffer_pct must be in (0, 1], got {self.drawdown_buffer_pct}"
            )

    @property
    def margin_budget(self) -> float:
        """Total committable margin = ``buffer_pct × equity``."""
        return self.drawdown_buffer_pct * self.account_equity

    @property
    def committed_margin(self) -> float:
        return self._committed_margin

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.margin_budget - self._committed_margin)

    def size_bet(
        self,
        raw_size: float,
        asset_class: AssetClass,
        *,
        commit: bool = True,
    ) -> SizedBet:
        """Convert a raw bet size to a margin commitment under all constraints.

        Parameters
        ----------
        raw_size
            Bet size in ``[0, 1]`` from the bet-sizer.
        asset_class
            Drives the ESMA leverage / margin fraction.
        commit
            When True (default), the resulting margin is added to the running
            committed total (so subsequent calls see less remaining budget).
            Set False for a dry-run "what-if" sizing.

        Returns
        -------
        :class:`SizedBet`.
        """
        if not 0.0 <= raw_size <= 1.0:
            raise ValueError(f"raw_size must be in [0, 1], got {raw_size}")

        scaled = raw_size / self.c95
        margin_fraction = margin_fraction_for(asset_class)
        # Margin a full-equity, fully-scaled position would require.
        desired_margin = scaled * margin_fraction * self.account_equity

        remaining = self.remaining_budget
        clamped = desired_margin > remaining
        margin = min(desired_margin, remaining)

        if commit:
            self._committed_margin += margin

        return SizedBet(
            raw_size=raw_size,
            scaled_size=scaled,
            margin=margin,
            clamped=clamped,
        )

    def reset(self) -> None:
        """Clear committed margin (e.g. at the start of a new trading session)."""
        self._committed_margin = 0.0

    def rehydrate_committed_margin(self, margin: float) -> None:
        """Seed the committed-margin total from the broker's open positions.

        AFML 0-8 final audit V4: on a restart the in-memory committed-margin
        total resets to 0, but the broker may still hold open positions
        consuming margin. Without rehydration the engine would size new bets
        against the *full* drawdown buffer and over-commit, breaching
        FTMO / ESMA limits. The execution engine calls this at startup with
        the summed margin of ``broker.get_open_positions()`` so the budget
        reflects reality before any new signal is processed.
        """
        if margin < 0.0:
            raise ValueError(f"rehydrated margin must be ≥ 0, got {margin}")
        self._committed_margin = margin
