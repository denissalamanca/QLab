"""Data-scaled hyperparameter grids per primary-alpha family (Ops M1.2).

Each family is a **2-D ordinal lattice** (clean neighborhoods for the plateau
selector, §M1.1) sized to **≥ `DSR_MIN_TRIALS` (30)** configs per (asset,
family) cohort so the Deflated Sharpe Ratio is never auto-quarantined.

Values are **relative / data-scaled**, never absolute price levels (anti-bias
rule): thresholds are multiples of the realized EWM σ (applied per asset at
runtime), windows are bar counts. Axes are listed **conservative-ascending** so
the plateau tie-break's parsimony preference (higher ordinal) is meaningful.

CEO-approved grids (2026-05-20):
- CUSUM     : vol_span(5) × threshold_mult(6) = 30
- Bollinger : window(6)   × num_std(5)        = 30
- Donchian  : window(6)   × pt_mult(5)        = 30
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from afml.research.plateau import Coord

#: Triple-barrier params not on a family's sweep axis (vol-scaled defaults).
DEFAULT_PT_MULT: float = 2.0
DEFAULT_SL_MULT: float = 2.0
DEFAULT_VERTICAL_BARS: int = 20


@dataclass(frozen=True, slots=True)
class GridAxis:
    """One hyperparameter axis: ordinal-ordered values, optionally integer-valued."""

    name: str
    values: tuple[float, ...]
    integer: bool = False

    def at(self, idx: int) -> float:
        v = self.values[idx]
        return int(v) if self.integer else v


@dataclass(frozen=True, slots=True)
class FamilyGrid:
    """A family's 2-D sweep lattice + the mapping coord → hyperparameter config."""

    family: str
    axis0: GridAxis
    axis1: GridAxis

    @property
    def dims(self) -> int:
        return 2

    @property
    def n_configs(self) -> int:
        return len(self.axis0.values) * len(self.axis1.values)

    def coords(self) -> list[Coord]:
        return [
            (i, j) for i, j in product(range(len(self.axis0.values)), range(len(self.axis1.values)))
        ]

    def config(self, coord: Coord) -> dict[str, float]:
        """Ordinal coord → ``{axis_name: value}`` (data-scaled multipliers / bar counts)."""
        if len(coord) != self.dims:
            raise ValueError(f"coord {coord} has wrong dims for family {self.family!r}")
        return {self.axis0.name: self.axis0.at(coord[0]), self.axis1.name: self.axis1.at(coord[1])}


CUSUM_GRID = FamilyGrid(
    "cusum",
    GridAxis("vol_span", (20.0, 35.0, 50.0, 75.0, 100.0), integer=True),
    GridAxis("threshold_mult", (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)),
)
BOLLINGER_GRID = FamilyGrid(
    "bollinger",
    GridAxis("window", (20.0, 35.0, 50.0, 75.0, 100.0, 150.0), integer=True),
    GridAxis("num_std", (1.5, 2.0, 2.5, 3.0, 3.5)),
)
DONCHIAN_GRID = FamilyGrid(
    "donchian",
    GridAxis("window", (20.0, 35.0, 50.0, 75.0, 100.0, 150.0), integer=True),
    GridAxis("pt_mult", (1.0, 1.5, 2.0, 2.5, 3.0)),
)

FAMILY_GRIDS: dict[str, FamilyGrid] = {
    g.family: g for g in (CUSUM_GRID, BOLLINGER_GRID, DONCHIAN_GRID)
}


def get_family_grid(family: str) -> FamilyGrid:
    """Look up a family's grid. Raises ``KeyError`` for an unknown family."""
    try:
        return FAMILY_GRIDS[family]
    except KeyError as e:
        raise KeyError(f"unknown alpha family {family!r}; known: {sorted(FAMILY_GRIDS)}") from e
