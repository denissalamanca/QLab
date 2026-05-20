"""Per-asset, config-independent precompute (Ops M1.3).

The bar frame, FFD ``d*``, and bar close are identical across every
hyperparameter config for an asset, so we build them once and reuse them across
the whole sweep — the dominant speed win.

**Bar granularity is derived from the holding regime** (see
``docs/specs/M1_bar_granularity.md``), not from an unconstrained JB minimisation
(which is degenerate — its infimum is the finest possible bar). We target the
regime's bar count ``B* = T/Δ`` and run the JB bar-type tournament **only over
candidates at that granularity**, so JB chooses the most-Gaussian *admissible*
bar (Pillar I regularised). The event count is then bounded by the CUSUM
first-passage law (Pillar II).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import polars as pl

from afml.config.assets import AssetSpec, get_asset
from afml.data import find_optimal_d, load_ticks
from afml.data.bars.selector import select_bar_type_streaming
from afml.data.bars.time_bars import build_time_bars
from afml.research.regimes import DEFAULT_REGIME, HoldingRegime

_MS_PER_HOUR: float = 3_600_000.0


@dataclass(frozen=True, slots=True)
class AssetPrecompute:
    """Config-independent per-asset artifacts, cached for the whole sweep."""

    asset: str
    bars: pl.DataFrame
    bar_type: str
    bar_parameter: float | str
    bar_jarque_bera: float
    n_bars: int
    regime_name: str
    bar_hours: float
    vertical_bars: int
    target_bar_count: int
    ffd_d: float | None
    ffd_window: int | None
    ffd_adf_pvalue: float | None


def precompute_asset(
    asset: AssetSpec | str,
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    regime: HoldingRegime = DEFAULT_REGIME,
) -> AssetPrecompute:
    """Load ticks → regime-granular bars (JB-selected) → FFD ``d*``.

    Parameters
    ----------
    asset
        An :class:`AssetSpec` or symbol from the universe.
    start, end
        Research window bounds (default: the asset's full on-disk history).
    regime
        The holding regime fixing the economic timescale (default: day-trading).
        Bar duration ``Δ = regime.bar_hours`` and the JB tournament is restricted
        to candidates at that granularity.
    """
    spec = asset if isinstance(asset, AssetSpec) else get_asset(asset)
    # Stay lazy: a full-history asset (e.g. BTCUSD 2020-2025 ≈ 360M ticks) cannot
    # be materialised on a workstation. The row count is read from parquet
    # metadata (no scan) and every bar candidate is built through the streaming
    # path, so peak memory is bounded by one tick slice — not the whole history.
    ticks = load_ticks(spec, start=start, end=end)
    n_ticks = int(ticks.select(pl.len()).collect().item())

    # Δ from the regime; build time bars at Δ to read the realised (active) bar
    # count, then size the information-bar candidates to match that count so the
    # JB tournament compares like-for-like granularity.
    interval_minutes = max(1, round(regime.bar_hours * 60.0))
    interval = f"{interval_minutes}m"
    time_bars = build_time_bars(ticks, interval=interval, streaming=True)
    b_target = max(time_bars.height, 1)
    expected_ticks = max(float(n_ticks) / b_target, 2.0)

    selection = select_bar_type_streaming(
        ticks,
        time_intervals=(interval,),
        tick_expected_ticks=(expected_ticks,),
    )
    bars = selection.bars

    close = bars["close"].to_numpy().astype(np.float64)
    ffd = find_optimal_d(close)

    return AssetPrecompute(
        asset=spec.symbol,
        bars=bars,
        bar_type=str(selection.winner.bar_type),
        bar_parameter=selection.winner.parameter,
        bar_jarque_bera=float(selection.winner.jarque_bera),
        n_bars=bars.height,
        regime_name=regime.name,
        bar_hours=regime.bar_hours,
        vertical_bars=regime.vertical_bars,
        target_bar_count=b_target,
        ffd_d=ffd.d_optimal,
        ffd_window=ffd.window_length,
        ffd_adf_pvalue=ffd.adf_pvalue_at_optimum,
    )
