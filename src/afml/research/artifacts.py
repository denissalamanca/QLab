"""Research-run artifacts + evidence sheet (Ops M1.7).

Serialises a :class:`SweepCertification` (plus the per-asset
:class:`AssetPrecompute` metadata) to a JSON-safe dict on disk — the resumable,
auditable record of one ``(asset, family)`` run — and renders the
``RESEARCH_RUN.md`` evidence sheet the roadmap mandates: per run, the bar
JB stat, event count, plateau coordinate + its worst-neighbor score, Brier vs.
naive baseline, surviving-cluster count, PBO and DSR.

JSON has no ``±inf``/tuple-key support, so coords serialise to ``[i, j]`` lists
and non-finite floats to ``null`` (an invalid surface point, a quarantined DSR).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from afml.research.precompute import AssetPrecompute
from afml.research.sweep import CertificationResult, SweepCertification, SweepResult

_DASH = "—"


def _num(value: float | None) -> float | None:
    """JSON-safe float: ``None`` for ``None`` or any non-finite value."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _fmt(value: float | None, places: int = 4) -> str:
    return _DASH if value is None else f"{value:.{places}f}"


def _certification_to_dict(cert: CertificationResult) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": cert.status,
        "passed": cert.passed,
        "n_events": cert.n_events,
        "n_surviving_features": len(cert.surviving_features),
        "surviving_features": list(cert.surviving_features),
        "n_trials": cert.n_trials,
        "detail": cert.detail,
        "pbo": None,
        "dsr": None,
        "dsr_quarantined": None,
        "target_shuffling_pvalue": None,
    }
    if cert.validation is not None:
        v = cert.validation
        out["pbo"] = _num(v.pbo.pbo)
        out["dsr"] = _num(v.dsr.dsr)
        out["dsr_quarantined"] = bool(v.dsr.quarantined)
        out["target_shuffling_pvalue"] = _num(v.target_shuffling.pvalue)
    return out


def _sweep_to_dict(sweep: SweepResult) -> dict[str, Any]:
    surface = [
        {
            "coord": list(t.coord),
            "config": dict(t.config),
            "status": t.status,
            "n_events": t.n_events,
            "objective": _num(t.objective),
            "valid": t.valid,
            "score": _num(t.surface_score),
            "experiment_id": str(t.experiment_id) if t.experiment_id is not None else None,
        }
        for t in sweep.trials
    ]
    plateau = {
        "selected": list(sweep.plateau.selected) if sweep.plateau.selected is not None else None,
        "robustness": _num(sweep.plateau.robustness),
        "plateau_size": sweep.plateau.plateau_size,
        "reason": sweep.plateau.reason,
    }
    winner: dict[str, Any] | None = None
    if sweep.winner_trial is not None:
        w = sweep.winner_trial
        winner = {
            "coord": list(w.coord),
            "config": dict(w.config),
            "experiment_id": str(w.experiment_id) if w.experiment_id is not None else None,
            "n_events": w.n_events,
            "objective": _num(w.objective),
            "brier_calibrated": _num(w.brier_calibrated),
            "brier_naive": _num(w.brier_naive),
        }
    return {
        "n_configs": len(sweep.trials),
        "n_valid": sweep.n_valid,
        "plateau": plateau,
        "winner": winner,
        "surface": surface,
    }


def run_to_dict(
    sc: SweepCertification,
    *,
    pc: AssetPrecompute | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Flatten a :class:`SweepCertification` (+ precompute metadata) to a JSON-safe dict."""
    record: dict[str, Any] = {
        "schema": "afml.research.run/1",
        "generated_at": datetime.now(UTC).isoformat(),
        "asset": sc.sweep.asset,
        "family": sc.sweep.family,
        "window": {"start": start, "end": end},
        "regime": None,
        "bars": None,
        "sweep": _sweep_to_dict(sc.sweep),
        "certification": (
            _certification_to_dict(sc.certification) if sc.certification is not None else None
        ),
    }
    if pc is not None:
        record["regime"] = {
            "name": pc.regime_name,
            "bar_hours": pc.bar_hours,
            "vertical_bars": pc.vertical_bars,
            "target_bar_count": pc.target_bar_count,
        }
        record["bars"] = {
            "type": pc.bar_type,
            "parameter": pc.bar_parameter,
            "jarque_bera": _num(pc.bar_jarque_bera),
            "n_bars": pc.n_bars,
            "ffd_d": _num(pc.ffd_d),
            "ffd_window": pc.ffd_window,
            "ffd_adf_pvalue": _num(pc.ffd_adf_pvalue),
        }
    return record


def write_run(
    sc: SweepCertification,
    path: str | Path,
    *,
    pc: AssetPrecompute | None = None,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    """Serialise a run to ``path`` (creating parent dirs). Returns the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = run_to_dict(sc, pc=pc, start=start, end=end)
    target.write_text(json.dumps(record, indent=2, sort_keys=False), encoding="utf-8")
    return target


def read_run(path: str | Path) -> dict[str, Any]:
    """Load a run artifact written by :func:`write_run`."""
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def _summary_row(run: Mapping[str, Any]) -> str:
    sweep = run.get("sweep", {})
    plateau = sweep.get("plateau", {})
    winner = sweep.get("winner")
    bars = run.get("bars")
    cert = run.get("certification")

    selected = plateau.get("selected")
    coord = _DASH if selected is None else f"({selected[0]},{selected[1]})"
    worst_neighbor = _fmt(plateau.get("robustness"))
    jb = _fmt(bars.get("jarque_bera"), 2) if bars else _DASH
    if cert is not None:
        events = str(cert["n_events"])
    elif winner is not None:
        events = str(winner["n_events"])
    else:
        events = _DASH
    brier = (
        f"{_fmt(winner['brier_calibrated'])}/{_fmt(winner['brier_naive'])}"
        if winner is not None
        else _DASH
    )
    clusters = str(cert["n_surviving_features"]) if cert else _DASH
    pbo = _fmt(cert.get("pbo")) if cert else _DASH
    dsr = _fmt(cert.get("dsr")) if cert else _DASH
    status = cert["status"] if cert else plateau.get("reason", "no plateau")
    return (
        f"| {run.get('asset', '?')} | {run.get('family', '?')} | {jb} | {events} | "
        f"{coord} | {worst_neighbor} | {brier} | {clusters} | {pbo} | {dsr} | {status} |"
    )


def render_research_run_md(
    runs: Sequence[Mapping[str, Any]],
    *,
    title: str = "AFML Research Run — M1 Evidence Sheet",
) -> str:
    """Render the ``RESEARCH_RUN.md`` evidence sheet from run artifacts."""
    lines: list[str] = [
        f"# {title}",
        "",
        f"_Generated {datetime.now(UTC).isoformat()} — {len(runs)} run(s)._",
        "",
        "Stable-plateau selection (Neighborhood-Minimax, §M1.1) — never the peak. "
        "`Worst-Nbr` is the selected centre's worst ±1 neighbor OOS Sharpe `R(g*)`; "
        "`Brier` is calibrated/naive (a survivor beats naive); PBO < 0.05 and DSR > 0.95 "
        "are the Phase-6 gate.",
        "",
        "| Asset | Family | Bar JB | Events | Plateau | Worst-Nbr | Brier (cal/naive) "
        "| Clusters | PBO | DSR | Status |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(_summary_row(run) for run in runs)

    survivors = [
        run for run in runs if (cert := run.get("certification")) is not None and cert.get("passed")
    ]
    lines.extend(["", f"## Certified survivors ({len(survivors)})", ""])
    if not survivors:
        lines.append(
            "_No certified survivors — every run was rejected, halted at MDA, or found "
            "no stable plateau. A valid, expected research outcome (never a manufactured "
            "strategy)._"
        )
    for run in survivors:
        cert = run["certification"]
        winner = run.get("sweep", {}).get("winner") or {}
        bars = run.get("bars") or {}
        regime = run.get("regime") or {}
        lines.extend([
            f"### {run['asset']} · {run['family']} · exp `{winner.get('experiment_id', '?')}`",
            "",
            f"- **Config:** `{winner.get('config', {})}`",
            f"- **Regime:** {regime.get('name', '?')} "
            f"(Δ={regime.get('bar_hours', '?')}h, V={regime.get('vertical_bars', '?')} bars)",
            f"- **Bars:** {bars.get('type', '?')} `{bars.get('parameter', '?')}` "
            f"· JB={_fmt(bars.get('jarque_bera'), 2)} · n={bars.get('n_bars', '?')} "
            f"· FFD d*={_fmt(bars.get('ffd_d'), 3)} (ADF p={_fmt(bars.get('ffd_adf_pvalue'))})",
            f"- **Events:** {cert['n_events']} · **Surviving clusters:** "
            f"{cert['n_surviving_features']}",
            f"- **Brier:** {_fmt(winner.get('brier_calibrated'))} (cal) vs "
            f"{_fmt(winner.get('brier_naive'))} (naive)",
            f"- **Plateau:** centre {run['sweep']['plateau']['selected']} · "
            f"worst-neighbor R={_fmt(run['sweep']['plateau']['robustness'])} · "
            f"size {run['sweep']['plateau']['plateau_size']}",
            f"- **Phase 6:** PBO={_fmt(cert.get('pbo'))} · DSR={_fmt(cert.get('dsr'))} · "
            f"target-shuffling p={_fmt(cert.get('target_shuffling_pvalue'))} · K={cert['n_trials']}",
            "",
        ])
    return "\n".join(lines) + "\n"
