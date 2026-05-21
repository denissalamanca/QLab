"""``afml research`` CLI (Ops M1.7) — drive the sweep, re-select, report.

Mounted under the operator CLI as ``afml research …``. Three verbs:

- ``sweep ASSET [FAMILY]`` — precompute the asset, run the surface sweep +
  Phase-6 certification for one family (or all three), and persist a JSON run
  artifact per ``(asset, family)``. Every config is logged to the Alpha Registry.
- ``select ARTIFACT`` — re-run the plateau selector on a persisted surface with
  a different ``--s-floor`` / ``--delta`` (cheap re-tuning; no re-sweep).
- ``report`` — render the ``RESEARCH_RUN.md`` evidence sheet from run artifacts.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from afml.config.settings import get_settings
from afml.core.registry import AlphaRegistryRepository
from afml.research.artifacts import read_run, render_research_run_md, write_run
from afml.research.diagnostics import render_diagnostics
from afml.research.grids import FAMILY_GRIDS, get_family_grid
from afml.research.plateau import Coord, select_plateau
from afml.research.precompute import precompute_asset
from afml.research.regimes import get_regime
from afml.research.sweep import sweep_and_certify

research_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="AFML research harness — sweep, plateau-select, certify, report.",
)


def _parse_date(value: str | None) -> date | None:
    return None if value is None else date.fromisoformat(value)


def _runs_base(out: Path | None) -> Path:
    return out if out is not None else get_settings().artifact_root / "research"


@research_app.command("sweep")
def sweep(
    asset: Annotated[str, typer.Argument(help="Asset symbol, e.g. EURUSD.")],
    family: Annotated[str, typer.Argument(help="Alpha family, or 'all' for every family.")] = "all",
    start: Annotated[str | None, typer.Option(help="Research window start (YYYY-MM-DD).")] = None,
    end: Annotated[str | None, typer.Option(help="Research window end (YYYY-MM-DD).")] = None,
    regime: Annotated[str, typer.Option(help="Holding regime (scalp/day/swing/position).")] = "day",
    out: Annotated[Path | None, typer.Option(help="Artifact base dir (default: settings).")] = None,
    s_floor: Annotated[float, typer.Option(help="Minimum worst-neighbor Sharpe.")] = 0.0,
    include_xgboost: Annotated[
        bool, typer.Option(help="Add XGBoost to the certification cohort.")
    ] = True,
    report: Annotated[
        bool, typer.Option(help="Also (re)render RESEARCH_RUN.md after the sweep.")
    ] = True,
) -> None:
    """Sweep ``(asset, family)`` end-to-end and persist run artifacts."""
    settings = get_settings()
    registry = AlphaRegistryRepository(
        settings.registry_db_url, wal_mode=settings.registry_wal_mode
    )
    registry.create_all()

    families = list(FAMILY_GRIDS) if family == "all" else [family]
    for fam in families:
        get_family_grid(fam)  # validate early
    base = _runs_base(out)

    hold = get_regime(regime)
    typer.echo(f"Precomputing {asset} (regime={regime}, Δ={hold.bar_hours}h) …")
    pc = precompute_asset(asset, start=_parse_date(start), end=_parse_date(end), regime=hold)
    typer.echo(
        f"  bars={pc.n_bars} type={pc.bar_type} JB={pc.bar_jarque_bera:.2f} "
        f"V={pc.vertical_bars} FFD d*={pc.ffd_d}"
    )

    for fam in families:
        typer.echo(f"Sweeping {asset}·{fam} ({get_family_grid(fam).n_configs} configs) …")
        result = sweep_and_certify(
            pc, fam, registry=registry, s_floor=s_floor, include_xgboost=include_xgboost
        )
        path = write_run(result, base / asset / f"{fam}.json", pc=pc, start=start, end=end)
        sel = result.sweep.plateau.selected
        if result.certification is None:
            typer.echo(f"  no stable plateau ({result.sweep.plateau.reason}) → {path}")
        else:
            c = result.certification
            typer.echo(
                f"  plateau {sel} R={result.sweep.plateau.robustness:.4f} → "
                f"{c.status} ({c.detail}) → {path}"
            )

    if report:
        _render_report(base)


@research_app.command("select")
def select(
    artifact: Annotated[Path, typer.Argument(help="Path to a run artifact JSON.")],
    s_floor: Annotated[float, typer.Option(help="Minimum worst-neighbor Sharpe.")] = 0.0,
    delta: Annotated[float, typer.Option(help="Plateau level tolerance.")] = 0.1,
) -> None:
    """Re-run the plateau selector on a persisted surface (cheap re-tuning, no re-sweep)."""
    run = read_run(artifact)
    family = str(run["family"])
    dims = get_family_grid(family).dims
    surface: dict[Coord, float] = {}
    for point in run["sweep"]["surface"]:
        coord: Coord = tuple(point["coord"])
        score = point["score"]
        surface[coord] = float("-inf") if score is None else float(score)

    result = select_plateau(surface, dims=dims, s_floor=s_floor, delta=delta)
    typer.echo(f"{run['asset']}·{family}: {result.reason}")
    typer.echo(f"  selected={result.selected} R={result.robustness:.4f} size={result.plateau_size}")


@research_app.command("report")
def report(
    runs_dir: Annotated[
        Path | None, typer.Option(help="Directory of run artifacts (default: settings).")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Output path for RESEARCH_RUN.md.")] = None,
) -> None:
    """Render the RESEARCH_RUN.md evidence sheet from persisted run artifacts."""
    base = _runs_base(runs_dir)
    path = _render_report(base, out=out)
    typer.echo(f"Wrote {path}")


def _render_report(base: Path, *, out: Path | None = None) -> Path:
    runs = [read_run(p) for p in sorted(base.rglob("*.json"))]
    target = out if out is not None else base / "RESEARCH_RUN.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_research_run_md(runs), encoding="utf-8")
    return target


@research_app.command("diagnose")
def diagnose(
    runs_dir: Annotated[
        Path | None, typer.Option(help="Directory of run artifacts (default: settings).")
    ] = None,
    asset: Annotated[str | None, typer.Option(help="Filter to one asset symbol.")] = None,
    family: Annotated[str | None, typer.Option(help="Filter to one alpha family.")] = None,
    out: Annotated[Path | None, typer.Option(help="Output path for DIAGNOSTICS.md.")] = None,
) -> None:
    """Per-stage observability: the attrition funnel, validity heatmaps, per-stage
    distributions, and feature-survival frequency — so you can see where the
    pipeline succeeds or degrades and steer next steps."""
    base = _runs_base(runs_dir)
    runs = [read_run(p) for p in sorted(base.rglob("*.json"))]
    if asset is not None:
        runs = [r for r in runs if r.get("asset") == asset]
    if family is not None:
        runs = [r for r in runs if r.get("family") == family]
    if not runs:
        typer.echo(f"No run artifacts under {base} (matching the filters).")
        raise typer.Exit(code=1)

    report_md = render_diagnostics(runs)
    target = out if out is not None else base / "DIAGNOSTICS.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report_md, encoding="utf-8")
    typer.echo(report_md)
    typer.echo(f"\nWrote {target}")
