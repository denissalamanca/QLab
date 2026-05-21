"""Per-stage research diagnostics — turn the sweep from a black box into a funnel.

A sweep's end-to-end verdict ("0 certified") tells you *nothing* about *where* the
pipeline succeeds or degrades. This module reconstructs, from the run artifacts,
the **attrition funnel** through the six AFML stages and the per-stage health, so
a human can see what's optimal and what's underperforming and steer accordingly:

1. **events** — did the primary alpha fire ≥ the floor? (too tight → starved)
2. **labels** — balanced? holding horizon coherent with the regime?
3. **features → selection** — did any feature cluster survive Clustered MDA?
   (the empty-MDA breaker is where near-efficient markets die)
4. **meta-model** — did Brain-2 beat the naive baseline? OOS Sharpe + fold spread.
5. **plateau** — do the valid configs form a robust neighbourhood, or isolated spikes?

Renders to text/markdown (the CLI surface). The same artifact fields feed the
React dashboard in the follow-up. Sections that need the per-config
``diagnostics`` block degrade gracefully on pre-observability artifacts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from statistics import median
from typing import Any

from afml.research.grids import get_family_grid

_DASH = "—"


def _f(x: float | None, p: int = 2) -> str:
    return _DASH if x is None else f"{x:.{p}f}"


def _bar(frac: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, frac))
    return "█" * round(frac * width)


def _funnel_row(k: int, n: int, label: str, extra: str = "") -> str:
    frac = k / n if n else 0.0
    return f"   {label:<34} {k:>2}/{n}  {_bar(frac):<20} {frac * 100:4.0f}%  {extra}"


def _surface(run: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sweep = run.get("sweep") or {}
    surface: list[Mapping[str, Any]] = sweep.get("surface") or []
    return surface


def _scores(surface: Sequence[Mapping[str, Any]]) -> list[float]:
    return [pt["score"] for pt in surface if pt.get("score") is not None]


def render_funnel(runs: Sequence[Mapping[str, Any]]) -> str:
    """The 6-stage attrition funnel per (asset, family) — where configs are lost."""
    lines = ["## Stage funnel — where each (asset, family) loses configs", "", "```"]
    for run in runs:
        surface = _surface(run)
        n = len(surface)
        if n == 0:
            continue
        st = Counter(str(pt.get("status")) for pt in surface)
        invalid = st.get("invalid", 0)
        mda_halt = st.get("FAILED_AT_MDA", 0)
        completed = st.get("completed", 0)
        reached_mda = completed + mda_halt
        n_valid = (run.get("sweep") or {}).get("n_valid", 0)
        scores = _scores(surface)
        best = max(scores) if scores else None
        med = median(scores) if scores else None
        bars = run.get("bars") or {}
        lines.append(
            f"{run.get('asset', '?')}·{run.get('family', '?')}  "
            f"bars={bars.get('n_bars', '?')} type={bars.get('type', '?')} "
            f"JB={_f(bars.get('jarque_bera'), 0)} FFD_d*={_f(bars.get('ffd_d'), 2)}"
        )

        reason = str((run.get("sweep") or {}).get("plateau", {}).get("reason", ""))
        certified = 1 if ((run.get("certification") or {}).get("passed")) else 0
        lines.append(_funnel_row(n, n, "① configs swept"))
        lines.append(
            _funnel_row(
                reached_mda, n, "② events≥floor → labels/features", f"({invalid} too few events)"
            )
        )
        lines.append(
            _funnel_row(
                completed, n, "③ survived ONC + Clustered MDA", f"({mda_halt} halted: no signal)"
            )
        )
        lines.append(
            _funnel_row(
                n_valid,
                n,
                "④ Brain-2 beat naive (valid)",
                f"best Sharpe={_f(best)} median={_f(med)}",
            )
        )
        lines.append(_funnel_row(certified, n, "⑤ robust plateau → certified", reason))
        lines.append("")
    lines.append("```")
    return "\n".join(lines)


def render_validity_heatmaps(runs: Sequence[Mapping[str, Any]]) -> str:
    """Per-run validity grid — █ valid / · invalid — exposes spikes vs plateaus."""
    lines = ["## Validity heatmaps (█ = valid surface point, · = invalid)", ""]
    for run in runs:
        family = str(run.get("family"))
        try:
            grid = get_family_grid(family)
        except KeyError:
            continue
        n0, n1 = len(grid.axis0.values), len(grid.axis1.values)
        valid = [[False] * n1 for _ in range(n0)]
        for pt in _surface(run):
            i, j = pt["coord"]
            if 0 <= i < n0 and 0 <= j < n1:
                valid[i][j] = bool(pt.get("valid"))
        nv = sum(v for row in valid for v in row)
        lines.append(f"**{run.get('asset')}·{family}** — {nv}/{n0 * n1} valid")
        lines.append("```")
        lines.append(f"   rows={grid.axis0.name} (↓)   cols={grid.axis1.name} (→)")
        for i in range(n0):
            lines.append("   " + "".join("█" if valid[i][j] else "·" for j in range(n1)))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _completed_diags(run: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        pt["diagnostics"]
        for pt in _surface(run)
        if pt.get("status") == "completed" and pt.get("diagnostics")
    ]


def _has_diagnostics(runs: Sequence[Mapping[str, Any]]) -> bool:
    return any(_completed_diags(run) for run in runs)


def render_distributions(runs: Sequence[Mapping[str, Any]]) -> str:
    """Per-stage health across completed configs: label balance, holding, Sharpe, Brier."""
    lines = ["## Per-stage distributions (completed configs)", ""]
    if not _has_diagnostics(runs):
        lines.append(
            "_No per-config diagnostics in these artifacts (pre-observability run). "
            "Re-run the sweep to populate label balance, holding coherence, OOS-Sharpe "
            "spread, and Brier detail._"
        )
        return "\n".join(lines)
    lines += [
        "| Asset | Family | n | label P[y=1] | hold/target | beats-naive | "
        "OOS Sharpe med [min,max] | fold-spread med |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        diags = _completed_diags(run)
        if not diags:
            continue
        pos = [d["label_pos_rate"] for d in diags if d.get("label_pos_rate") is not None]
        ratios = [
            d["mean_holding_bars"] / d["target_holding_bars"]
            for d in diags
            if d.get("mean_holding_bars") is not None and d.get("target_holding_bars")
        ]
        beats = sum(
            1
            for d in diags
            if d.get("brier_calibrated") is not None
            and d.get("brier_naive") is not None
            and d["brier_calibrated"] < d["brier_naive"]
        )
        fold_meds = [median(d["fold_sharpes"]) for d in diags if d.get("fold_sharpes")]
        spreads = [
            max(d["fold_sharpes"]) - min(d["fold_sharpes"]) for d in diags if d.get("fold_sharpes")
        ]
        sh_med = median(fold_meds) if fold_meds else None
        sh_min = min(fold_meds) if fold_meds else None
        sh_max = max(fold_meds) if fold_meds else None
        lines.append(
            f"| {run.get('asset')} | {run.get('family')} | {len(diags)} | "
            f"{_f(median(pos) if pos else None)} | {_f(median(ratios) if ratios else None)} | "
            f"{beats}/{len(diags)} | "
            f"{_f(sh_med)} [{_f(sh_min)},{_f(sh_max)}] | {_f(median(spreads) if spreads else None)} |"
        )
    lines += [
        "",
        "_hold/target ≈ 1 means the realised holding matches the regime's vertical barrier; "
        "a wide fold-spread on a high median = one lucky fold (overfit), not a real edge._",
    ]
    return "\n".join(lines)


def render_feature_frequency(runs: Sequence[Mapping[str, Any]], *, top: int = 15) -> str:
    """Which features survive Clustered MDA most often — the 'what carries signal' view."""
    lines = ["## Feature survival frequency (across completed configs, per asset)", ""]
    if not _has_diagnostics(runs):
        lines.append(
            "_No per-config diagnostics in these artifacts — re-run the sweep to populate._"
        )
        return "\n".join(lines)
    by_asset: dict[str, Counter[str]] = {}
    totals: dict[str, int] = {}
    for run in runs:
        asset = str(run.get("asset"))
        diags = _completed_diags(run)
        if not diags:
            continue
        counter = by_asset.setdefault(asset, Counter())
        totals[asset] = totals.get(asset, 0) + len(diags)
        for d in diags:
            for feat in d.get("surviving_features", []):
                counter[feat] += 1
    for asset, counter in by_asset.items():
        n = totals.get(asset, 0)
        lines.append(f"**{asset}** — {n} completed configs; top {top} surviving features:")
        lines.append("```")
        if not counter:
            lines.append("   (no features survived MDA in any config)")
        for feat, cnt in counter.most_common(top):
            frac = cnt / n if n else 0.0
            lines.append(f"   {feat:<28} {cnt:>3}/{n}  {_bar(frac, 16):<16} {frac * 100:3.0f}%")
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def render_diagnostics(
    runs: Sequence[Mapping[str, Any]],
    *,
    title: str = "AFML Research Diagnostics — per-stage observability",
) -> str:
    """Compose the full diagnostics report (funnel + heatmaps + distributions + features)."""
    header = [
        f"# {title}",
        "",
        f"_Generated {datetime.now(UTC).isoformat()} — {len(runs)} run(s)._",
        "",
        "Read top-to-bottom: the **funnel** shows where configs are lost; the "
        "**heatmaps** show whether survivors cluster (plateau) or scatter (spikes); "
        "the **distributions** and **feature frequency** show per-stage health.",
        "",
    ]
    return "\n".join([
        *header,
        render_funnel(runs),
        "",
        render_validity_heatmaps(runs),
        "",
        render_distributions(runs),
        "",
        render_feature_frequency(runs),
        "",
    ])
