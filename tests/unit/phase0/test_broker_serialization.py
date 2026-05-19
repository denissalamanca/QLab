"""Phase 0-8 final audit V3 — numpy/pandas-safe message-bus serialization.

The inter-agent bus must survive ML artifacts that carry ``numpy`` scalars /
arrays and ``pandas.Timestamp`` values. ``encode_json`` / ``decode_json`` are
the canonical codec; these tests prove the round-trip the stdlib ``json``
module would crash on.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from afml.core.broker import decode_json, encode_json


@pytest.mark.phase0
def test_encode_numpy_scalars() -> None:
    """numpy float64 / int64 / bool_ serialize to native JSON values."""
    payload = {
        "prob": np.float64(0.8732),
        "count": np.int64(42),
        "flag": np.bool_(True),
    }
    restored = decode_json(encode_json(payload))
    assert restored["prob"] == pytest.approx(0.8732)
    assert restored["count"] == 42
    assert restored["flag"] is True
    # And the decoded types are native Python, not numpy.
    assert isinstance(restored["prob"], float)
    assert isinstance(restored["count"], int)


@pytest.mark.phase0
def test_encode_numpy_array() -> None:
    """A numpy prediction array (Phase 5 → Phase 7) serializes to a list."""
    arr = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    restored = decode_json(encode_json({"predictions": arr}))
    assert restored["predictions"] == pytest.approx([0.1, 0.5, 0.9])


@pytest.mark.phase0
def test_encode_int_array() -> None:
    arr = np.array([1, 2, 3], dtype=np.int64)
    restored = decode_json(encode_json({"labels": arr}))
    assert restored["labels"] == [1, 2, 3]


@pytest.mark.phase0
def test_encode_datetime_isoformat() -> None:
    """datetime-like values (incl. pandas.Timestamp) serialize via isoformat."""
    ts = dt.datetime(2026, 5, 20, 12, 30, 15, tzinfo=dt.UTC)
    restored = decode_json(encode_json({"t": ts}))
    # Round-trips to a parseable ISO-8601 string.
    assert dt.datetime.fromisoformat(restored["t"]) == ts


@pytest.mark.phase0
def test_encode_pandas_timestamp() -> None:
    """The audit names pandas.Timestamp explicitly — verify it round-trips."""
    pd = pytest.importorskip("pandas")
    ts = pd.Timestamp("2026-05-20T12:30:15Z")
    restored = decode_json(encode_json({"bar_time": ts}))
    assert pd.Timestamp(restored["bar_time"]) == ts


@pytest.mark.phase0
def test_encode_nested_numpy_in_dict() -> None:
    """A loosely-typed nested payload with mixed numpy values serializes."""
    payload = {
        "asset": "EURUSD",
        "metrics": {"sharpe": np.float64(1.4), "n": np.int64(252)},
        "z_scores": np.array([1.1, 2.2], dtype=np.float64),
    }
    restored = decode_json(encode_json(payload))
    assert restored["asset"] == "EURUSD"
    assert restored["metrics"]["sharpe"] == pytest.approx(1.4)
    assert restored["metrics"]["n"] == 252
    assert restored["z_scores"] == pytest.approx([1.1, 2.2])


@pytest.mark.phase0
def test_encode_rejects_truly_unserializable() -> None:
    """A genuinely unserializable object still raises TypeError (we don't
    silently swallow programmer errors)."""

    class Opaque:
        pass

    with pytest.raises(TypeError, match="not JSON serializable"):
        encode_json({"x": Opaque()})
