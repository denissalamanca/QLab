"""Phase 1 — cross-phase burn-in alignment helpers (AFML 0-4 audit V2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from afml.data.integration import (
    align_events_to_features,
    align_labels_to_features,
    feature_matrix_to_numpy,
)


def _bars_ts(n: int) -> list[datetime]:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return [t0 + timedelta(minutes=i) for i in range(n)]


@pytest.mark.phase1
def test_align_events_drops_burn_in_timestamps() -> None:
    """Events whose timestamps fall inside the Phase 3 burn-in window (and
    therefore are NOT present in the feature timestamp set) must be dropped."""
    ts = _bars_ts(200)
    # Features start at index 50 — simulates Phase 3 dropping the first 50
    # bars to the rolling-window burn-in.
    features = pl.DataFrame(
        {
            "timestamp": ts[50:],
            "feat_a": np.arange(150, dtype=np.float64),
        },
        schema={"timestamp": pl.Datetime("ms", "UTC"), "feat_a": pl.Float64},
    )
    events = pl.DataFrame(
        {
            "timestamp": ts[::20],  # 0, 20, 40, 60, 80, ...
            "side": ["long"] * 10,
        },
        schema={"timestamp": pl.Datetime("ms", "UTC"), "side": pl.Utf8},
    )
    aligned, report = align_events_to_features(events, features)
    # Three burn-in events at indices 0, 20, 40 must be dropped.
    assert report.n_events_in == 10
    assert report.n_events_out == 7
    assert report.n_dropped_burn_in == 3
    # The aligned frame's timestamps must all be present in the feature frame.
    feature_ts_set = set(features["timestamp"].to_numpy().tolist())
    for ts_value in aligned["timestamp"].to_numpy().tolist():
        assert ts_value in feature_ts_set


@pytest.mark.phase1
def test_align_labels_to_features_preserves_label_columns() -> None:
    """``align_labels_to_features`` reuses the same intersection but keys on
    ``event_timestamp`` and sorts the result chronologically."""
    ts = _bars_ts(100)
    features = pl.DataFrame(
        {"timestamp": ts[30:], "feat_a": np.arange(70, dtype=np.float64)},
        schema={"timestamp": pl.Datetime("ms", "UTC"), "feat_a": pl.Float64},
    )
    labels = pl.DataFrame(
        {
            "event_timestamp": [ts[10], ts[40], ts[20], ts[50]],
            "label": [1, 0, 1, 0],
            "side": ["long", "short", "long", "short"],
        },
        schema={
            "event_timestamp": pl.Datetime("ms", "UTC"),
            "label": pl.Int64,
            "side": pl.Utf8,
        },
    )
    aligned, report = align_labels_to_features(labels, features)
    assert report.n_dropped_burn_in == 2  # ts[10] and ts[20] are below the cutoff
    # Sorted chronologically.
    out_ts = aligned["event_timestamp"].to_numpy().tolist()
    assert out_ts == sorted(out_ts)
    # Label / side columns preserved.
    assert set(aligned.columns) == {"event_timestamp", "label", "side"}


@pytest.mark.phase1
def test_alignment_handles_dtype_precision_mismatch() -> None:
    """An events frame with us-precision timestamps and a feature frame at
    ms must still intersect cleanly (dtype-aware cast under the hood)."""
    ts = _bars_ts(60)
    features = pl.DataFrame(
        {"timestamp": ts, "feat_a": np.zeros(60, dtype=np.float64)},
        schema={"timestamp": pl.Datetime("ms", "UTC"), "feat_a": pl.Float64},
    )
    events = pl.DataFrame(
        {"timestamp": [ts[5], ts[10]], "side": ["long", "short"]},
        schema={"timestamp": pl.Datetime("us", "UTC"), "side": pl.Utf8},
    )
    _aligned, report = align_events_to_features(events, features)
    assert report.n_events_out == 2  # both survive


@pytest.mark.phase1
def test_alignment_rejects_missing_columns() -> None:
    features = pl.DataFrame({"ts": [1, 2, 3]})
    events = pl.DataFrame({"event_t": [1]})
    with pytest.raises(KeyError, match="timestamp"):
        align_events_to_features(events, features)
    with pytest.raises(KeyError, match="timestamp"):
        align_events_to_features(events, features, event_timestamp_col="event_t")


@pytest.mark.phase1
def test_feature_matrix_to_numpy_rejects_non_finite() -> None:
    """A NaN row that escaped Phase 3 must NOT be silently passed to Phase 4 —
    callers should be forced to fix their alignment upstream."""
    features = pl.DataFrame({
        "timestamp": [1, 2, 3],
        "feat_a": [1.0, float("nan"), 3.0],
    })
    with pytest.raises(ValueError, match="non-finite"):
        feature_matrix_to_numpy(features)


@pytest.mark.phase1
def test_feature_matrix_to_numpy_round_trip_preserves_order() -> None:
    features = pl.DataFrame({
        "timestamp": [1, 2, 3, 4],
        "feat_a": [1.0, 2.0, 3.0, 4.0],
        "feat_b": [10.0, 20.0, 30.0, 40.0],
    })
    X, names = feature_matrix_to_numpy(features)
    assert names == ["feat_a", "feat_b"]
    assert X.shape == (4, 2)
    np.testing.assert_array_equal(X[:, 0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(X[:, 1], [10.0, 20.0, 30.0, 40.0])
