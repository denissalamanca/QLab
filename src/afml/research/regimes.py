"""Holding-period regimes — the configurable strategy timescale (Ops M1).

The research bar granularity is **derived** from a target holding horizon, not
hard-coded (see ``docs/specs/M1_bar_granularity.md``). A :class:`HoldingRegime`
states the strategy's *economic* timescale; the bar duration, vertical barrier,
and target bar count then follow from the **first-passage law**:

- A trade exits at the first touch of the ``±pt·σ`` barrier, whose expected
  first-passage time is ``pt²`` bars ⇒ ``mean_hold ≈ pt² · Δ`` ⇒ ``Δ = mean_hold / pt²``.
- The vertical barrier caps the *max* hold: ``V = round(max_hold / Δ)``.
- Over a window of trading-length ``T`` (hours), the target bar count is ``B* = T / Δ``.

**The default is day-trading, but the system is not constrained to it.** Agents
may sweep any set of regimes (scalp → position) to explore strategy types; each
regime is a distinct research dimension logged with the trial.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The ``pt_mult`` the mean-hold ↔ Δ mapping is anchored on (the grid's default
#: profit-take multiple). ``mean_hold ≈ pt_reference² · Δ``.
DEFAULT_PT_REFERENCE: float = 2.0


@dataclass(frozen=True, slots=True)
class HoldingRegime:
    """A configurable strategy timescale.

    Parameters
    ----------
    name
        Short identifier (logged into the trial's hyperparameter vector).
    mean_hold_hours
        Target *mean* trade duration (first-passage to the profit/stop barrier).
    max_hold_hours
        Target *max* hold (the vertical barrier). Must be ≥ ``mean_hold_hours``.
    pt_reference
        The profit-take multiple anchoring ``mean ≈ pt²·Δ`` (default 2.0).
    """

    name: str
    mean_hold_hours: float
    max_hold_hours: float
    pt_reference: float = DEFAULT_PT_REFERENCE

    def __post_init__(self) -> None:
        if self.mean_hold_hours <= 0.0:
            raise ValueError(f"mean_hold_hours must be > 0, got {self.mean_hold_hours}")
        if self.max_hold_hours < self.mean_hold_hours:
            raise ValueError(
                f"max_hold_hours ({self.max_hold_hours}) must be ≥ "
                f"mean_hold_hours ({self.mean_hold_hours})"
            )
        if self.pt_reference <= 0.0:
            raise ValueError(f"pt_reference must be > 0, got {self.pt_reference}")

    @property
    def bar_hours(self) -> float:
        """Bar duration Δ = mean_hold / pt² (the first-passage anchor)."""
        return self.mean_hold_hours / (self.pt_reference**2)

    @property
    def vertical_bars(self) -> int:
        """Vertical-barrier horizon in bars: round(max_hold / Δ)."""
        return max(1, round(self.max_hold_hours / self.bar_hours))

    def target_bar_count(self, span_hours: float) -> int:
        """Target number of bars B* over a window of ``span_hours`` trading-time."""
        return max(1, int(span_hours / self.bar_hours))


#: Built-in regimes (open set — callers may construct their own).
REGIMES: dict[str, HoldingRegime] = {
    "scalp": HoldingRegime("scalp", mean_hold_hours=1.0, max_hold_hours=4.0),
    "day": HoldingRegime("day", mean_hold_hours=6.0, max_hold_hours=24.0),
    "swing": HoldingRegime("swing", mean_hold_hours=120.0, max_hold_hours=480.0),
    "position": HoldingRegime("position", mean_hold_hours=480.0, max_hold_hours=2000.0),
}

#: Default research regime (CEO-approved: day-trading).
DEFAULT_REGIME: HoldingRegime = REGIMES["day"]


def get_regime(name: str) -> HoldingRegime:
    """Look up a built-in regime by name. Raises ``KeyError`` if unknown."""
    try:
        return REGIMES[name]
    except KeyError as e:
        raise KeyError(f"unknown regime {name!r}; built-ins: {sorted(REGIMES)}") from e
