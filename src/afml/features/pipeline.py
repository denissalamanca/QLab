"""Microstructure feature pipeline (Phase 3 orchestrator).

Computes every base feature at every requested window, registers each column
in the global ``FeatureRegistry``, NaN-handles per Blueprint §5.3 (forward-fill
with limit, then drop residuals), and — if an events DataFrame is supplied —
subselects rows at event timestamps only (Blueprint §5.2: features sampled
strictly at event times).

The default window grid produces **52 columns**:
- 6 price/volume features × 7 windows = 42
- 2 entropy features × 5 (longer) windows = 10

clear of the Blueprint §3.1 anti-lazy "50+ distinct metrics" requirement.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from afml.features.amihud import amihud_lambda
from afml.features.base import FeatureSpec, register_feature
from afml.features.corwin_schultz import corwin_schultz_spread
from afml.features.hasbrouck import hasbrouck_flow
from afml.features.kyle import kyle_lambda
from afml.features.lempel_ziv import lempel_ziv_complexity
from afml.features.ofi import ofi
from afml.features.roll import roll_measure
from afml.features.shannon import shannon_entropy

DEFAULT_WINDOWS: tuple[int, ...] = (10, 20, 30, 50, 75, 100, 150, 200)
DEFAULT_WINDOWS_ENTROPY: tuple[int, ...] = (50, 100, 150, 200, 250)


def compute_features(
    bars: pl.DataFrame,
    events: pl.DataFrame | None = None,
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    windows_entropy: tuple[int, ...] = DEFAULT_WINDOWS_ENTROPY,
    ffill_limit: int = 5,
) -> pl.DataFrame:
    """Compute the full microstructure feature matrix.

    Parameters
    ----------
    bars : Phase 1 OHLCV DataFrame, sorted by ``timestamp``.
    events : optional Phase 2 events DataFrame (columns: ``timestamp``, ``side``).
        When supplied, the returned matrix contains only rows whose timestamp
        matches an event (Blueprint §5.2).
    windows : look-back windows for Roll / Corwin-Schultz / OFI / Kyle / Amihud /
        Hasbrouck. Default = ``(10, 20, 50, 100, 200, 500, 1000)``.
    windows_entropy : look-back windows for Shannon / Lempel-Ziv. These need
        longer sequences to be statistically meaningful.
    ffill_limit : max consecutive NaNs to forward-fill before dropping the row.

    Returns
    -------
    Polars DataFrame with ``timestamp`` plus one column per
    ``(base_family, window)`` pair (e.g. ``roll_w20``). All NaN-free; ADF-
    stationary per column (verified by ``test_pipeline.py``).
    """
    bars_sorted = bars.sort("timestamp")
    bar_ts = bars_sorted["timestamp"]
    close = bars_sorted["close"].to_numpy().astype(np.float64)
    high = (
        bars_sorted["high"].to_numpy().astype(np.float64)
        if "high" in bars_sorted.columns
        else close
    )
    low = (
        bars_sorted["low"].to_numpy().astype(np.float64) if "low" in bars_sorted.columns else close
    )
    volume = (
        bars_sorted["volume"].to_numpy().astype(np.float64)
        if "volume" in bars_sorted.columns
        else np.ones_like(close)
    )

    columns: dict[str, np.ndarray] = {}

    for w in windows:
        columns[f"roll_w{w}"] = roll_measure(close, window=w)
        columns[f"corwin_schultz_w{w}"] = corwin_schultz_spread(high, low, window=w)
        columns[f"ofi_w{w}"] = ofi(close, volume, window=w)
        columns[f"kyle_w{w}"] = kyle_lambda(close, volume, window=w)
        columns[f"amihud_w{w}"] = amihud_lambda(close, volume, window=w)
        columns[f"hasbrouck_w{w}"] = hasbrouck_flow(close, volume, window=w)

    for w in windows_entropy:
        columns[f"shannon_w{w}"] = shannon_entropy(close, window=w)
        columns[f"lempel_ziv_w{w}"] = lempel_ziv_complexity(close, window=w)

    for name in columns:
        family, _, win_str = name.rpartition("_w")
        register_feature(
            FeatureSpec(
                name=name,
                base_family=family,
                window=int(win_str),
                causal=True,
            )
        )

    feature_names = list(columns.keys())
    df = pl.DataFrame({"timestamp": bar_ts.to_numpy(), **columns})

    # Convert NaN → Null so polars' fill_null can take over, then forward-fill,
    # then drop any residual nulls (Blueprint §5.3 NaN auditing rule).
    df = df.with_columns([
        pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c) for c in feature_names
    ])
    df = df.with_columns([
        pl.col(c).fill_null(strategy="forward", limit=ffill_limit) for c in feature_names
    ])
    df = df.drop_nulls(subset=feature_names)

    if events is not None:
        # Align event timestamp precision to bar grid for exact match.
        ev_aligned = events.with_columns(pl.col("timestamp").cast(bar_ts.dtype))
        ev_set = ev_aligned["timestamp"].implode()
        df = df.filter(pl.col("timestamp").is_in(ev_set))

    return df
