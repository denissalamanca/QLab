"""Cross-phase integration helpers (AFML 0-4 audit V2).

Phase 3's microstructure features carry a **burn-in window** equal to
``max(window) − 1`` bars — rolling statistics over an N-bar window simply do
not produce a value at the head. Phase 2's CUSUM filter, by contrast, fires
as soon as cumulative deviation crosses the volatility threshold, which can
happen during the burn-in period.

If left unmanaged, two integration failures result:

1. **NaN contamination.** Naïvely forward-filling NaN feature rows for the
   burn-in events injects garbage data into Brain 2 — features that were
   "imputed from the future" are non-causal in a way unit tests on a single
   module won't catch.
2. **Length mismatch.** Phase 3 silently sub-selects feature rows at event
   timestamps that survive its ``drop_nulls`` pass. The Phase 2 labels
   DataFrame keeps every event. The two arrive at Phase 4 with mismatched
   row counts, blowing up the indexer at runtime — *if you're lucky*; if
   you're not, ``polars`` may silently broadcast and produce a wrong answer.

This module exposes :func:`align_events_to_features` and
:func:`align_labels_to_features` — the canonical filters that downstream
Phase 4 / Phase 5 callers must use to bridge Phase 2 ↔ Phase 3.

The functions are minimal: an inner-join by timestamp, with dtype-aware
casting so that microsecond-precision events line up with millisecond
bars. They are intentionally *not* implicit in :func:`compute_features` —
the caller owns the alignment so the dropped events surface in the audit
log instead of vanishing silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Audit log returned alongside the aligned frames.

    Attributes
    ----------
    n_events_in
        Row count of the input events frame.
    n_events_out
        Row count after intersection with the feature timestamps.
    n_dropped_burn_in
        Events whose timestamp did NOT survive Phase 3's ``drop_nulls`` —
        almost always concentrated at the head of the bar series.
    """

    n_events_in: int
    n_events_out: int
    n_dropped_burn_in: int

    @property
    def drop_rate(self) -> float:
        return self.n_dropped_burn_in / self.n_events_in if self.n_events_in else 0.0


def align_events_to_features(
    events: pl.DataFrame,
    features: pl.DataFrame,
    *,
    event_timestamp_col: str = "timestamp",
    feature_timestamp_col: str = "timestamp",
) -> tuple[pl.DataFrame, AlignmentReport]:
    """Inner-join an events frame against the feature matrix by timestamp.

    Drops every event whose timestamp does not appear in the feature
    matrix's timestamp column — typically the head events that fell inside
    the Phase 3 rolling-window burn-in.

    Parameters
    ----------
    events
        Phase 2 events frame (or any frame keyed by event time). Must
        contain ``event_timestamp_col``.
    features
        Phase 3 feature matrix. Must contain ``feature_timestamp_col``.
    event_timestamp_col, feature_timestamp_col
        Column names; defaults match the Brain 1 events / Phase 3 features
        conventions used elsewhere in the lab.

    Returns
    -------
    ``(aligned_events, report)`` — the filtered events frame plus an
    :class:`AlignmentReport` documenting the burn-in drop count.
    """
    if event_timestamp_col not in events.columns:
        raise KeyError(f"events frame missing column {event_timestamp_col!r}")
    if feature_timestamp_col not in features.columns:
        raise KeyError(f"features frame missing column {feature_timestamp_col!r}")

    feature_ts = features[feature_timestamp_col]
    ts_dtype = feature_ts.dtype

    # Cast event timestamps to the feature dtype so different precisions
    # (us vs ms) don't kill the intersection silently.
    events_cast = events.with_columns(pl.col(event_timestamp_col).cast(ts_dtype))
    feature_ts_set = feature_ts.implode()
    aligned = events_cast.filter(pl.col(event_timestamp_col).is_in(feature_ts_set))

    n_in = int(events.height)
    n_out = int(aligned.height)
    return aligned, AlignmentReport(
        n_events_in=n_in,
        n_events_out=n_out,
        n_dropped_burn_in=n_in - n_out,
    )


def align_labels_to_features(
    labels_df: pl.DataFrame,
    features: pl.DataFrame,
    *,
    label_timestamp_col: str = "event_timestamp",
    feature_timestamp_col: str = "timestamp",
) -> tuple[pl.DataFrame, AlignmentReport]:
    """Inner-join a Triple-Barrier labels frame against the feature matrix.

    Same contract as :func:`align_events_to_features` but uses the
    ``event_timestamp`` column conventional to ``TripleBarrierLabels.df``.

    Returned frame is sorted by ``event_timestamp`` so the row order matches
    the feature matrix (which is timestamp-sorted by Phase 3).
    """
    aligned, report = align_events_to_features(
        labels_df,
        features,
        event_timestamp_col=label_timestamp_col,
        feature_timestamp_col=feature_timestamp_col,
    )
    aligned = aligned.sort(label_timestamp_col)
    return aligned, report


def feature_matrix_to_numpy(
    features: pl.DataFrame,
    *,
    timestamp_col: str = "timestamp",
) -> tuple[np.ndarray, list[str]]:
    """Convenience: drop the timestamp column and return ``(X, feature_names)``.

    Strictly raises if any NaN / Inf survived Phase 3 — that would mean
    the burn-in alignment was skipped or feature pipeline misbehaved.
    """
    feature_columns = [c for c in features.columns if c != timestamp_col]
    X = features.select(feature_columns).to_numpy().astype(np.float64)
    if not np.all(np.isfinite(X)):
        bad_count = int((~np.isfinite(X)).sum())
        raise ValueError(
            f"feature matrix has {bad_count} non-finite values — did you forget "
            f"to align events to features after Phase 3 (AFML 0-4 audit V2)?"
        )
    return X, feature_columns
