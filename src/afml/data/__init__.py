"""Phase 1 — Structural data engineering.

Public API:
- ``load_ticks`` — Polars LazyFrame loader for Dukascopy parquet files.
- ``build_time_bars`` / ``build_tick_imbalance_bars`` / ``build_tick_run_bars`` —
  Information-driven bar generators.
- ``select_bar_type`` — Jarque-Bera-minimization selector with stable-plateau check.
- ``ffd_weights`` / ``ffd_apply`` / ``find_optimal_d`` — Fixed-Width Fractional
  Differencing.
- ``adf_pvalue`` / ``jarque_bera_statistic`` — stationarity / normality wrappers.
- ``truncation_hash`` / ``assert_no_leakage`` — anti-leakage gates.
"""

from afml.data.bars.selector import (
    BarSelection,
    BarTypeSweepResult,
    select_bar_type,
)
from afml.data.bars.tick_imbalance import build_tick_imbalance_bars
from afml.data.bars.tick_run import build_tick_run_bars
from afml.data.bars.time_bars import build_time_bars
from afml.data.causality import assert_no_leakage, truncation_hash
from afml.data.ffd import (
    FFDResult,
    ffd_apply,
    ffd_weights,
    find_optimal_d,
)
from afml.data.ingest import load_ticks
from afml.data.stationarity import adf_pvalue, jarque_bera_statistic

__all__ = [
    "BarSelection",
    "BarTypeSweepResult",
    "FFDResult",
    "adf_pvalue",
    "assert_no_leakage",
    "build_tick_imbalance_bars",
    "build_tick_run_bars",
    "build_time_bars",
    "ffd_apply",
    "ffd_weights",
    "find_optimal_d",
    "jarque_bera_statistic",
    "load_ticks",
    "select_bar_type",
    "truncation_hash",
]
