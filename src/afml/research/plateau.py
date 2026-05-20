"""Neighborhood-Minimax plateau selection (Ops M1.1 — the anti-curve-fit rule).

The research sweep scores a hyperparameter grid; a sharp performance *spike* is
almost always overfit (a one-step parameter nudge collapses it), while a broad
*plateau* is robust. We select the grid point whose entire ±1 neighborhood is
strong, by maximising the **worst-neighbor** score:

    R(g) = min( s(g), min over neighbors h of s(h) )

`argmax R` can never be an isolated spike (a spike has a weak neighbor that
drags its `R` down). The criterion has **no free tuning constant** — so the
selector itself cannot be meta-overfit — and it is a literal statement of
robustness: *"still strong if the market nudges my parameters one grid step."*

Design choices for our application:
- **Full-neighborhood eligibility.** Only *interior* points whose entire ±1
  Chebyshev neighborhood is present and valid can be plateau centres. A point on
  the grid boundary is ineligible: you haven't sampled one side, so you cannot
  claim robustness — extend the grid instead (the sweep warns when the raw
  optimum sits on the boundary).
- **No-plateau rejection.** If the best `R` is below ``s_floor`` (default 0 — a
  positive worst-neighbor objective) we return ``selected=None``: *there is no
  stable configuration*. A valid, expected research outcome, never a
  manufactured strategy. (The minimax `R` already rejects spikes, so the
  connected ``plateau_size`` is reported as a diagnostic + used only to
  tie-break, never to veto a high-`R` point.)
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

#: A grid point in ordinal lattice coordinates (one int per hyperparameter axis).
Coord = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PlateauResult:
    """Outcome of :func:`select_plateau`.

    Attributes
    ----------
    selected
        The chosen plateau-centre coordinate, or ``None`` when no stable
        configuration exists.
    robustness
        ``R(g*)`` — the worst-neighbor objective at the winner (``-inf`` if none).
    plateau_size
        Size of the connected region (within ``delta`` of the winner's level)
        containing the winner — a diagnostic of plateau breadth (1 ⇒ a lone
        peak at its own level, though it may still be robust by ``R``).
    reason
        Human-readable explanation of the decision.
    """

    selected: Coord | None
    robustness: float
    plateau_size: int
    reason: str


def _neighbors(g: Coord, dims: int) -> list[Coord]:
    """The up-to ``3**dims - 1`` Chebyshev-distance-1 neighbors of ``g``."""
    out: list[Coord] = []
    for offset in product((-1, 0, 1), repeat=dims):
        if any(offset):
            out.append(tuple(g[i] + offset[i] for i in range(dims)))
    return out


def _connected_size(
    start: Coord, level: float, scores: Mapping[Coord, float], dims: int, delta: float
) -> int:
    """Size of the connected component of ``{h: s(h) >= level - delta}`` at ``start``."""
    threshold = level - delta
    seen: set[Coord] = {start}
    queue: deque[Coord] = deque([start])
    while queue:
        cur = queue.popleft()
        for h in _neighbors(cur, dims):
            s = scores.get(h, float("-inf"))
            if h not in seen and math.isfinite(s) and s >= threshold:
                seen.add(h)
                queue.append(h)
    return len(seen)


def select_plateau(
    scores: Mapping[Coord, float],
    *,
    dims: int,
    s_floor: float = 0.0,
    delta: float = 0.1,
    tie_eps: float = 1e-9,
) -> PlateauResult:
    """Select the robust plateau centre of a scored hyperparameter grid.

    Parameters
    ----------
    scores
        Maps each grid coordinate (``dims``-tuple of ints) to its objective
        ``s(g)``. Invalid configurations map to ``-inf`` (or are absent).
    dims
        Number of hyperparameter axes (lattice dimensionality).
    s_floor
        Minimum acceptable ``R(g*)``; below it ⇒ no stable config.
    delta
        Level tolerance defining a point's "plateau" (connected near-level region).
    tie_eps
        Two robustness values within this are treated as tied.

    Returns
    -------
    :class:`PlateauResult`.
    """
    if dims < 1:
        raise ValueError(f"dims must be ≥ 1, got {dims}")

    finite = {g: s for g, s in scores.items() if math.isfinite(s)}
    if not finite:
        return PlateauResult(None, float("-inf"), 0, "no valid configurations")

    required_neighbors = 3**dims - 1  # full interior neighborhood
    robustness: dict[Coord, float] = {}
    for g, s in finite.items():
        nb = [finite[h] for h in _neighbors(g, dims) if h in finite]
        if len(nb) < required_neighbors:
            continue  # boundary point — cannot confirm a plateau, ineligible
        robustness[g] = min(s, *nb)

    if not robustness:
        return PlateauResult(
            None, float("-inf"), 0, "no eligible interior point (grid too small / all on boundary)"
        )

    r_max = max(robustness.values())
    candidates = [g for g, r in robustness.items() if r >= r_max - tie_eps]

    # Tie-break: largest connected plateau, then parsimony (higher ordinal sum —
    # grids list values conservative-ascending), then lexicographically smallest.
    def _key(g: Coord) -> tuple[int, int, tuple[int, ...]]:
        size = _connected_size(g, finite[g], finite, dims, delta)
        return (size, sum(g), tuple(-c for c in g))

    winner = max(candidates, key=_key)
    r_star = robustness[winner]
    plateau_size = _connected_size(winner, finite[winner], finite, dims, delta)

    if r_star < s_floor:
        return PlateauResult(
            None, r_star, plateau_size, f"best worst-neighbor score {r_star:.4f} < floor {s_floor}"
        )

    return PlateauResult(winner, r_star, plateau_size, "stable plateau selected")
