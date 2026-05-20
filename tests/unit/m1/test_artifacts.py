"""M1.7 — run artifacts (JSON round-trip) + RESEARCH_RUN.md rendering + CLI."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from typer.testing import CliRunner

from afml.research.artifacts import (
    read_run,
    render_research_run_md,
    run_to_dict,
    write_run,
)
from afml.research.cli import research_app
from afml.research.grids import get_family_grid
from afml.research.harness import TrialResult
from afml.research.plateau import Coord, PlateauResult
from afml.research.sweep import (
    STATUS_CERTIFIED,
    CertificationResult,
    SweepCertification,
    SweepResult,
)
from afml.validation import (
    DSRResult,
    PBOResult,
    TargetShufflingResult,
    ValidationResult,
)

pytestmark = pytest.mark.m1

WINNER: Coord = (2, 2)


def _trial(coord: Coord, *, objective: float | None, valid: bool) -> TrialResult:
    return TrialResult(
        family="cusum",
        coord=coord,
        config={"vol_span": 50.0, "threshold_mult": 1.5},
        status="completed" if valid else "FAILED_AT_MDA",
        n_events=900,
        objective=objective,
        valid=valid,
        brier_calibrated=0.2280 if valid else None,
        brier_naive=0.2490 if valid else None,
        experiment_id=uuid4(),
        detail="stub",
    )


def _passing_validation() -> ValidationResult:
    return ValidationResult(
        pbo=PBOResult(pbo=0.02, n_splits=15, n_strategies=5, logits=np.zeros(1, dtype=np.float64)),
        dsr=DSRResult(
            dsr=0.991,
            sharpe_observed=1.6,
            expected_max_sharpe=0.3,
            n_trials=30,
            n_observations=900,
            quarantined=False,
        ),
        target_shuffling=TargetShufflingResult(
            brier_real=0.228,
            brier_shuffled=np.array([0.25, 0.26], dtype=np.float64),
            pvalue=0.001,
            n_shuffles=30,
        ),
        incumbent_idx=0,
    )


def _sweep_result(*, with_plateau: bool) -> SweepResult:
    winner = _trial(WINNER, objective=1.2, valid=True)
    trials = [winner, _trial((0, 0), objective=None, valid=False)]
    surface = {t.coord: t.surface_score for t in trials}
    if with_plateau:
        plateau = PlateauResult(WINNER, 0.95, 6, "stable plateau selected")
        winner_trial: TrialResult | None = winner
    else:
        plateau = PlateauResult(None, float("-inf"), 0, "no valid configurations")
        winner_trial = None
    return SweepResult("EURUSD", "cusum", tuple(trials), surface, plateau, winner_trial)


def _full_grid_sweep_cert() -> SweepCertification:
    """A full CUSUM grid with a 3×3 valid block around WINNER (re-selectable surface)."""
    grid = get_family_grid("cusum")
    block = {(i, j) for i in (1, 2, 3) for j in (1, 2, 3)}
    trials = [
        _trial(coord, objective=1.0 if coord in block else None, valid=coord in block)
        for coord in grid.coords()
    ]
    surface = {t.coord: t.surface_score for t in trials}
    winner = next(t for t in trials if t.coord == WINNER)
    plateau = PlateauResult(WINNER, 1.0, 9, "stable plateau selected")
    sweep = SweepResult("EURUSD", "cusum", tuple(trials), surface, plateau, winner)
    cert = CertificationResult(
        asset="EURUSD",
        family="cusum",
        coord=WINNER,
        experiment_id=winner.experiment_id,
        n_events=900,
        surviving_features=("roll_measure",),
        n_trials=30,
        status=STATUS_CERTIFIED,
        validation=_passing_validation(),
        detail="ok",
    )
    return SweepCertification(sweep, cert)


def _sweep_cert(*, with_plateau: bool, passed: bool) -> SweepCertification:
    sweep = _sweep_result(with_plateau=with_plateau)
    if not with_plateau:
        return SweepCertification(sweep, None)
    validation = _passing_validation()
    cert = CertificationResult(
        asset="EURUSD",
        family="cusum",
        coord=WINNER,
        experiment_id=sweep.winner_trial.experiment_id if sweep.winner_trial else None,
        n_events=900,
        surviving_features=("roll_measure", "kyle_lambda"),
        n_trials=30,
        status=STATUS_CERTIFIED if passed else "rejected",
        validation=validation,
        detail="PBO=0.0200 DSR=0.9910",
    )
    return SweepCertification(sweep, cert)


# ----------------------------------------------------------------- serialization


def test_run_to_dict_is_json_safe() -> None:
    record = run_to_dict(_sweep_cert(with_plateau=True, passed=True), start="2020-01-01", end="x")
    text = json.dumps(record)  # would raise on inf / tuple keys
    assert "Infinity" not in text  # -inf surface scores must serialise to null
    # Invalid surface point serialises to null score.
    invalid = next(p for p in record["sweep"]["surface"] if not p["valid"])
    assert invalid["score"] is None
    # Coords are lists, not tuples.
    assert record["sweep"]["plateau"]["selected"] == [2, 2]


def test_write_read_round_trip(tmp_path: Path) -> None:
    sc = _sweep_cert(with_plateau=True, passed=True)
    path = write_run(sc, tmp_path / "EURUSD" / "cusum.json", start="2020-01-01", end="2025-12-31")
    assert path.exists()
    loaded = read_run(path)
    assert loaded["asset"] == "EURUSD"
    assert loaded["certification"]["pbo"] == pytest.approx(0.02)
    assert loaded["certification"]["dsr"] == pytest.approx(0.991)
    assert loaded["window"] == {"start": "2020-01-01", "end": "2025-12-31"}


def test_report_lists_certified_survivor() -> None:
    record = run_to_dict(_sweep_cert(with_plateau=True, passed=True))
    md = render_research_run_md([record])
    assert "Certified survivors (1)" in md
    assert "EURUSD" in md and "cusum" in md
    assert "0.0200" in md  # PBO
    assert "0.9910" in md  # DSR
    assert "(2,2)" in md  # plateau coord in the summary table


def test_report_empty_and_no_survivors() -> None:
    assert "0 run(s)" in render_research_run_md([])
    rejected = run_to_dict(_sweep_cert(with_plateau=True, passed=False))
    md = render_research_run_md([rejected])
    assert "Certified survivors (0)" in md
    assert "No certified survivors" in md


def test_report_no_plateau_row() -> None:
    record = run_to_dict(_sweep_cert(with_plateau=False, passed=False))
    md = render_research_run_md([record])
    assert "no valid configurations" in md
    assert "Certified survivors (0)" in md


# -------------------------------------------------------------------------- CLI


def test_cli_report_writes_file(tmp_path: Path) -> None:
    write_run(_sweep_cert(with_plateau=True, passed=True), tmp_path / "EURUSD" / "cusum.json")
    out = tmp_path / "RESEARCH_RUN.md"
    result = CliRunner().invoke(
        research_app, ["report", "--runs-dir", str(tmp_path), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Certified survivors (1)" in out.read_text(encoding="utf-8")


def test_cli_select_reruns_plateau(tmp_path: Path) -> None:
    artifact = write_run(_full_grid_sweep_cert(), tmp_path / "cusum.json")
    result = CliRunner().invoke(research_app, ["select", str(artifact), "--s-floor", "0.0"])
    assert result.exit_code == 0, result.output
    assert "selected=(2, 2)" in result.output  # the 3×3 block centre is re-selected


def test_cli_select_rejects_with_high_floor(tmp_path: Path) -> None:
    artifact = write_run(_full_grid_sweep_cert(), tmp_path / "cusum.json")
    result = CliRunner().invoke(research_app, ["select", str(artifact), "--s-floor", "99.0"])
    assert result.exit_code == 0, result.output
    assert "selected=None" in result.output  # R(g*)=1.0 < floor 99 → no stable config
