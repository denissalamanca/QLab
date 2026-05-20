"""M1.1 — Neighborhood-Minimax plateau selector (Ops roadmap §M1.1 DoD)."""

from __future__ import annotations

import pytest

from afml.research import select_plateau
from afml.research.plateau import Coord

pytestmark = pytest.mark.m1


def _grid_2d(n: int, fill: float) -> dict[Coord, float]:
    return {(i, j): fill for i in range(n) for j in range(n)}


# ---------------------------------------------------- spike loses to plateau
def test_spike_loses_to_plateau() -> None:
    """A tall narrow spike must never beat a broad plateau (the core DoD)."""
    scores = _grid_2d(5, 0.5)
    scores[(1, 1)] = 3.0  # isolated spike — neighbors stay 0.5
    for i in (2, 3, 4):  # a 3x3 plateau at ~2.0; (3,3) is its fully-interior centre
        for j in (2, 3, 4):
            scores[(i, j)] = 2.0

    result = select_plateau(scores, dims=2)

    assert result.selected == (3, 3)
    assert result.robustness == pytest.approx(2.0)
    assert result.selected != (1, 1)


# ------------------------------------------------------------- monotone ramp
def test_monotone_ramp_picks_high_interior_not_endpoint() -> None:
    scores: dict[Coord, float] = {(i,): float(i) for i in range(5)}  # [0,1,2,3,4]
    result = select_plateau(scores, dims=1)
    # Endpoint (4,) is on the boundary (only one neighbor) ⇒ ineligible;
    # the highest *interior* point is (3,).
    assert result.selected == (3,)
    assert result.robustness == pytest.approx(2.0)


# -------------------------------------------------------------- flat surface
def test_flat_surface_returns_interior_deterministic() -> None:
    scores = _grid_2d(3, 1.0)  # only (1,1) is interior in a 3x3
    first = select_plateau(scores, dims=2)
    second = select_plateau(scores, dims=2)
    assert first.selected == (1, 1)
    assert first == second  # deterministic


# ------------------------------------------------- no stable config (spikes)
def test_all_spikes_returns_none() -> None:
    """Alternating high/low → every point has a bad neighbor → no plateau."""
    scores: dict[Coord, float] = {(i,): (5.0 if i % 2 == 0 else -5.0) for i in range(5)}
    result = select_plateau(scores, dims=1)
    assert result.selected is None
    assert result.robustness < 0.0


def test_plateau_below_floor_rejected() -> None:
    scores = _grid_2d(3, 1.0)
    result = select_plateau(scores, dims=2, s_floor=2.0)  # plateau at 1.0 < floor
    assert result.selected is None


# ----------------------------------------------------------- boundary guard
def test_boundary_spike_not_selected() -> None:
    scores = _grid_2d(3, 1.0)
    scores[(0, 0)] = 10.0  # huge — but it's a corner (boundary), ineligible
    result = select_plateau(scores, dims=2)
    assert result.selected == (1, 1)
    assert result.selected != (0, 0)


def test_invalid_neighbor_makes_point_ineligible() -> None:
    """Full-neighborhood rule: one -inf neighbor disqualifies the only interior point."""
    scores = _grid_2d(3, 1.0)
    scores[(0, 1)] = float("-inf")  # (1,1)'s neighbor now invalid → 7 valid < 8
    result = select_plateau(scores, dims=2)
    assert result.selected is None


def test_no_valid_configs_returns_none() -> None:
    scores: dict[Coord, float] = {(0,): float("-inf"), (1,): float("-inf"), (2,): float("-inf")}
    result = select_plateau(scores, dims=1)
    assert result.selected is None


# -------------------------------------------------------- order invariance
def test_selection_is_order_invariant() -> None:
    scores = _grid_2d(5, 0.5)
    scores[(1, 1)] = 3.0
    for i in (2, 3, 4):
        for j in (2, 3, 4):
            scores[(i, j)] = 2.0

    forward = select_plateau(scores, dims=2)
    reversed_scores = dict(reversed(list(scores.items())))
    backward = select_plateau(reversed_scores, dims=2)
    assert forward.selected == backward.selected
    assert forward.robustness == pytest.approx(backward.robustness)
