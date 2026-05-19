"""Microstructure feature pipeline (Phase 3 orchestrator).

Computes every base feature at every requested window, registers each column
in the global ``FeatureRegistry``, NaN-handles per Blueprint §5.3 (forward-fill
with limit, then drop residuals), and — if an events DataFrame is supplied —
subselects rows at event timestamps only (Blueprint §5.2: features sampled
strictly at event times).

**FFD stationarity rescue (AFML audit Vulnerability 3):** instead of dropping
a feature column whose ADF p-value exceeds ``adf_p_threshold``, the pipeline
routes it through ``find_optimal_d`` and replaces the column with its FFD'd
version when stationarity is recoverable. A column is only DROPPED if FFD
itself cannot achieve stationarity — which should be rare with a sensible
d-grid. This keeps the feature dimensionality high (≥ 50) per the Blueprint
DoD instead of starving it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from afml.data.ffd import find_optimal_d
from afml.data.stationarity import adf_pvalue
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


@dataclass(frozen=True, slots=True)
class StationarityRescueReport:
    """Audit log of the FFD stationarity-rescue step (AFML audit V3)."""

    columns_passed_adf_directly: list[str]
    columns_rescued_via_ffd: dict[str, float]  # name → optimal d
    columns_dropped: list[str]  # FFD also failed

    @property
    def n_total(self) -> int:
        return (
            len(self.columns_passed_adf_directly)
            + len(self.columns_rescued_via_ffd)
            + len(self.columns_dropped)
        )

    @property
    def n_rescued(self) -> int:
        return len(self.columns_rescued_via_ffd)

    @property
    def n_dropped(self) -> int:
        return len(self.columns_dropped)


def _enforce_stationarity_via_ffd(
    df: pl.DataFrame,
    feature_names: list[str],
    *,
    adf_p_threshold: float = 0.05,
) -> tuple[pl.DataFrame, list[str], StationarityRescueReport]:
    """Replace ADF-failing columns with their FFD-rescued versions.

    AFML audit V3: lazily dropping non-stationary features starves the feature
    set. For each column, if ADF fails, search for an optimal ``d`` via FFD
    and substitute the FFD'd series. Drop only if FFD ALSO fails to achieve
    stationarity (rare with a sensible d-grid).

    Returns ``(df_out, surviving_names, report)``. ``surviving_names`` reflects
    columns that remain after any drops, and may be a strict subset of
    ``feature_names``. ``df_out``'s schema is updated accordingly.
    """
    passed: list[str] = []
    rescued: dict[str, float] = {}
    dropped: list[str] = []
    replaced: dict[str, np.ndarray] = {}

    for col in feature_names:
        values = df[col].to_numpy().astype(np.float64)
        # Constant columns can't be regressed; skip ADF — they're trivially
        # stationary in the degenerate sense.
        if np.std(values) == 0.0:
            passed.append(col)
            continue
        try:
            p = adf_pvalue(values)
        except ValueError:
            # Too few observations even for ADF — drop conservatively.
            dropped.append(col)
            continue
        if p < adf_p_threshold:
            passed.append(col)
            continue

        # FFD rescue.
        result = find_optimal_d(values)
        if result.d_optimal is None or result.series is None:
            dropped.append(col)
            continue
        rescued[col] = result.d_optimal
        # The FFD output is shorter than the input by exactly l*. We pad with
        # NaN at the head and let downstream forward-fill + drop_nulls clean up.
        padding = len(values) - len(result.series)
        padded = np.full(len(values), np.nan, dtype=np.float64)
        padded[padding:] = result.series
        replaced[col] = padded

    # Apply replacements.
    if replaced:
        df = df.with_columns([pl.Series(name=col, values=arr) for col, arr in replaced.items()])
    # Drop columns that even FFD couldn't rescue.
    if dropped:
        df = df.drop(dropped)
    surviving = [c for c in feature_names if c not in dropped]
    return (
        df,
        surviving,
        StationarityRescueReport(
            columns_passed_adf_directly=passed,
            columns_rescued_via_ffd=rescued,
            columns_dropped=dropped,
        ),
    )


def compute_features(
    bars: pl.DataFrame,
    events: pl.DataFrame | None = None,
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    windows_entropy: tuple[int, ...] = DEFAULT_WINDOWS_ENTROPY,
    ffill_limit: int = 5,
    enforce_stationarity: bool = True,
    adf_p_threshold: float = 0.05,
) -> pl.DataFrame:
    """Compute the full microstructure feature matrix.

    Returns the feature DataFrame only. For the FFD-rescue audit log call
    ``compute_features_with_report`` instead.
    """
    df, _ = _compute_features_internal(
        bars,
        events=events,
        windows=windows,
        windows_entropy=windows_entropy,
        ffill_limit=ffill_limit,
        enforce_stationarity=enforce_stationarity,
        adf_p_threshold=adf_p_threshold,
    )
    return df


def compute_features_with_report(
    bars: pl.DataFrame,
    events: pl.DataFrame | None = None,
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    windows_entropy: tuple[int, ...] = DEFAULT_WINDOWS_ENTROPY,
    ffill_limit: int = 5,
    adf_p_threshold: float = 0.05,
) -> tuple[pl.DataFrame, StationarityRescueReport]:
    """Same as ``compute_features`` but also returns the ``StationarityRescueReport``.

    Always runs the FFD rescue step (``enforce_stationarity=True``) — the
    report wouldn't be meaningful otherwise.
    """
    df, report = _compute_features_internal(
        bars,
        events=events,
        windows=windows,
        windows_entropy=windows_entropy,
        ffill_limit=ffill_limit,
        enforce_stationarity=True,
        adf_p_threshold=adf_p_threshold,
    )
    assert report is not None
    return df, report


def _compute_features_internal(
    bars: pl.DataFrame,
    events: pl.DataFrame | None = None,
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    windows_entropy: tuple[int, ...] = DEFAULT_WINDOWS_ENTROPY,
    ffill_limit: int = 5,
    enforce_stationarity: bool = True,
    adf_p_threshold: float = 0.05,
) -> tuple[pl.DataFrame, StationarityRescueReport | None]:
    """Compute the full microstructure feature matrix.

    Parameters
    ----------
    bars : Phase 1 OHLCV DataFrame with ``timestamp``, ``open``, ``high``,
        ``low``, ``close``, ``volume``. Sorted by ``timestamp``.
    events : optional Phase 2 events DataFrame (``timestamp``, ``side``). When
        supplied the result is restricted to event timestamps only.
    windows : look-back grid for price/volume features (Roll, Corwin-Schultz,
        OFI, Kyle, Amihud, Hasbrouck).
    windows_entropy : look-back grid for entropy-family features (Shannon, LZ).
    ffill_limit : forward-fill limit before dropping residual nulls.
    enforce_stationarity : if True (default), route any ADF-failing column
        through FFD before considering it a drop candidate (AFML audit V3).
    adf_p_threshold : significance threshold for the ADF stationarity check.
    return_rescue_report : if True, returns ``(df, report)`` tuple.

    Returns
    -------
    Polars DataFrame with ``timestamp`` plus one column per surviving
    ``(family, window)`` pair. NaN-free; every column is ADF-stationary
    directly or post-FFD.
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
    # Per-bar Δt in seconds (V2: time-adjusted Corwin-Schultz needs this).
    bar_durations = _bar_durations_seconds(bar_ts)

    columns: dict[str, np.ndarray] = {}

    for w in windows:
        columns[f"roll_w{w}"] = roll_measure(close, window=w)
        columns[f"corwin_schultz_w{w}"] = corwin_schultz_spread(
            high, low, bar_durations=bar_durations, window=w
        )
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

    # FFD stationarity rescue (AFML audit V3).
    rescue_report: StationarityRescueReport | None = None
    if enforce_stationarity:
        df, feature_names, rescue_report = _enforce_stationarity_via_ffd(
            df, feature_names, adf_p_threshold=adf_p_threshold
        )
        # Re-clean any NaN introduced by FFD padding. Forward-fill is limited;
        # the FFD warm-up tail is much longer, so the final ``drop_nulls`` will
        # uniformly trim the head across all rescued columns.
        df = df.with_columns([
            pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
            for c in feature_names
        ])
        df = df.with_columns([
            pl.col(c).fill_null(strategy="forward", limit=ffill_limit) for c in feature_names
        ])
        df = df.drop_nulls(subset=feature_names)

        # Final stationarity sweep — after the global trim, a previously-rescued
        # column may still fail ADF on the reduced slice (the trimmed length is
        # shorter than the FFD's original ADF check). Drop those holdouts so the
        # invariant "every surviving column passes ADF" holds end-to-end.
        final_dropped: list[str] = []
        for col in feature_names:
            values = df[col].to_numpy().astype(np.float64)
            if np.std(values) == 0.0:
                continue
            try:
                p = adf_pvalue(values)
            except ValueError:
                final_dropped.append(col)
                continue
            if p >= adf_p_threshold:
                final_dropped.append(col)
        if final_dropped:
            df = df.drop(final_dropped)
            final_dropped_set = set(final_dropped)
            feature_names = [c for c in feature_names if c not in final_dropped_set]
            # Move final-dropped columns out of whichever bucket they were in
            # (passed-directly or rescued-via-FFD) and into ``columns_dropped``.
            rescue_report = StationarityRescueReport(
                columns_passed_adf_directly=[
                    c
                    for c in rescue_report.columns_passed_adf_directly
                    if c not in final_dropped_set
                ],
                columns_rescued_via_ffd={
                    k: v
                    for k, v in rescue_report.columns_rescued_via_ffd.items()
                    if k not in final_dropped_set
                },
                columns_dropped=rescue_report.columns_dropped + final_dropped,
            )

    if events is not None:
        ev_aligned = events.with_columns(pl.col("timestamp").cast(bar_ts.dtype))
        ev_set = ev_aligned["timestamp"].implode()
        df = df.filter(pl.col("timestamp").is_in(ev_set))

    return df, rescue_report


def _bar_durations_seconds(timestamp_series: pl.Series) -> np.ndarray:
    """Per-bar duration in seconds — first bar inherits the second bar's Δt."""
    ts_ns = timestamp_series.cast(pl.Int64).to_numpy()
    if ts_ns.size < 2:
        return np.ones(ts_ns.size, dtype=np.float64)
    delta_ns = np.diff(ts_ns)
    durations = np.empty(ts_ns.size, dtype=np.float64)
    durations[1:] = delta_ns.astype(np.float64) / 1e9
    durations[0] = durations[1] if durations.size > 1 else 1.0
    return durations
