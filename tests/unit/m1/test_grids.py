"""M1.2 — data-scaled hyperparameter grids (≥30/cohort, 2-D lattice)."""

from __future__ import annotations

import pytest

from afml.research import FAMILY_GRIDS, get_family_grid
from afml.validation import DSR_MIN_TRIALS

pytestmark = pytest.mark.m1


def test_every_family_has_at_least_dsr_min_trials() -> None:
    assert set(FAMILY_GRIDS) == {"cusum", "bollinger", "donchian"}
    for family, grid in FAMILY_GRIDS.items():
        assert grid.dims == 2, family
        assert grid.n_configs >= DSR_MIN_TRIALS, f"{family} cohort < {DSR_MIN_TRIALS}"
        assert len(grid.coords()) == grid.n_configs


def test_config_maps_ordinal_coord_to_data_scaled_values() -> None:
    cusum = get_family_grid("cusum")
    cfg = cusum.config((0, 0))
    assert cfg == {"vol_span": 20, "threshold_mult": 0.5}
    # Integer axis returns a real int (so it can seed an EWM span / window).
    assert isinstance(cfg["vol_span"], int)
    top = cusum.config((4, 5))
    assert top == {"vol_span": 100, "threshold_mult": 3.0}


def test_coords_cover_full_lattice() -> None:
    grid = get_family_grid("bollinger")
    coords = grid.coords()
    assert (0, 0) in coords and (5, 4) in coords
    assert len(set(coords)) == grid.n_configs  # unique


def test_unknown_family_raises() -> None:
    with pytest.raises(KeyError):
        get_family_grid("ichimoku")
