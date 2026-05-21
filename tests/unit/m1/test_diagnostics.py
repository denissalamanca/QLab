"""M1 observability — per-stage diagnostics renderer (funnel, heatmap, distributions)."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from afml.research.artifacts import run_to_dict
from afml.research.diagnostics import (
    render_diagnostics,
    render_distributions,
    render_feature_frequency,
    render_funnel,
    render_validity_heatmaps,
)
from afml.research.grids import get_family_grid
from afml.research.harness import TrialDiagnostics, TrialResult
from afml.research.plateau import Coord, PlateauResult
from afml.research.sweep import SweepCertification, SweepResult

pytestmark = pytest.mark.m1


def _diag(*, completed: bool) -> TrialDiagnostics:
    base = TrialDiagnostics(
        n_alpha_events=900,
        n_events_modeled=850,
        label_pos_rate=0.47,
        return_mean=0.0008,
        return_std=0.011,
        mean_holding_bars=14.0,
        target_holding_bars=16,
        n_features_in=52,
    )
    if not completed:
        return replace(base, halted_at_mda=True, n_surviving_features=0)
    return replace(
        base,
        halted_at_mda=False,
        n_surviving_features=3,
        surviving_features=("roll_measure", "kyle_lambda", "ofi"),
        brier_calibrated=0.228,
        brier_naive=0.249,
        fold_sharpes=(0.5, 0.8, 0.3, 0.6, 0.7),
    )


def _trial(coord: Coord, *, status: str, with_diag: bool) -> TrialResult:
    completed = status == "completed"
    valid = completed
    diag = None
    if with_diag:
        diag = _diag(completed=completed) if status != "invalid" else TrialDiagnostics(900)
    return TrialResult(
        family="cusum",
        coord=coord,
        config={"vol_span": 50.0, "threshold_mult": 1.0},
        status=status,
        n_events=900,
        objective=0.6 if valid else None,
        valid=valid,
        brier_calibrated=0.228 if completed else None,
        brier_naive=0.249 if completed else None,
        experiment_id=uuid4() if status != "invalid" else None,
        detail="stub",
        diagnostics=diag,
    )


def _run_dict(*, with_diag: bool) -> dict[str, Any]:
    """A CUSUM run: a 3x3 valid block (completed) + scattered halts/invalids."""
    grid = get_family_grid("cusum")
    block = {(i, j) for i in (1, 2, 3) for j in (1, 2, 3)}
    trials = []
    for k, coord in enumerate(grid.coords()):
        if coord in block:
            status = "completed"
        elif k % 3 == 0:
            status = "FAILED_AT_MDA"
        else:
            status = "invalid"
        trials.append(_trial(coord, status=status, with_diag=with_diag))
    surface = {t.coord: t.surface_score for t in trials}
    plateau = PlateauResult(None, float("-inf"), 0, "no eligible interior point")
    sweep = SweepResult("EURUSD", "cusum", tuple(trials), surface, plateau, None)
    return run_to_dict(SweepCertification(sweep, None))


def test_funnel_shows_stages_and_attrition() -> None:
    md = render_funnel([_run_dict(with_diag=True)])
    assert "Stage funnel" in md
    assert "① configs swept" in md
    assert "③ survived ONC + Clustered MDA" in md
    assert "EURUSD·cusum" in md
    assert "█" in md  # progress bars rendered


def test_heatmap_marks_valid_and_invalid() -> None:
    md = render_validity_heatmaps([_run_dict(with_diag=True)])
    assert "Validity heatmaps" in md
    assert "█" in md and "·" in md
    assert "rows=vol_span" in md  # axis labels from the grid


def test_distributions_populated_with_diagnostics() -> None:
    md = render_distributions([_run_dict(with_diag=True)])
    assert "label P[y=1]" in md
    assert "OOS Sharpe med" in md
    assert "EURUSD" in md and "cusum" in md


def test_distributions_graceful_without_diagnostics() -> None:
    md = render_distributions([_run_dict(with_diag=False)])
    assert "No per-config diagnostics" in md
    assert "Re-run the sweep" in md


def test_feature_frequency_ranks_surviving_features() -> None:
    md = render_feature_frequency([_run_dict(with_diag=True)])
    assert "roll_measure" in md  # appears in every completed config → top of the list
    assert "EURUSD" in md


def test_feature_frequency_graceful_without_diagnostics() -> None:
    md = render_feature_frequency([_run_dict(with_diag=False)])
    assert "No per-config diagnostics" in md


def test_render_diagnostics_composes_all_sections() -> None:
    md = render_diagnostics([_run_dict(with_diag=True)])
    for section in (
        "Stage funnel",
        "Validity heatmaps",
        "Per-stage distributions",
        "Feature survival frequency",
    ):
        assert section in md


def test_diagnostics_serialization_round_trips_numpy() -> None:
    # diagnostics carry np scalars in practice; the artifact must stay JSON-safe.
    record = _run_dict(with_diag=True)
    json.dumps(record)  # must not raise
    completed = [pt for pt in record["sweep"]["surface"] if pt["status"] == "completed"]
    assert completed and completed[0]["diagnostics"]["n_surviving_features"] == 3


def test_empty_diagnostics_for_invalid_trial() -> None:
    t = _trial((0, 0), status="invalid", with_diag=True)
    assert t.diagnostics is not None
    assert t.diagnostics.n_alpha_events == 900
    assert t.diagnostics.halted_at_mda is None  # never reached selection


def test_fold_sharpe_arrays_numpy_safe() -> None:
    d = TrialDiagnostics(900, fold_sharpes=tuple(np.array([0.5, 0.6], dtype=np.float64)))
    assert len(d.fold_sharpes) == 2
