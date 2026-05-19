"""Phase 1 — Dukascopy parquet loader."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from afml.data.ingest import load_ticks, verify_schema


@pytest.fixture
def tick_parquet(tmp_path: Path) -> Path:
    """Write a small synthetic Dukascopy-schema parquet to a tmp path."""
    n = 1000
    timestamps = [datetime(2024, 1, 1, tzinfo=UTC).replace(microsecond=i * 1000) for i in range(n)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "bid": [1.10 + i * 1e-5 for i in range(n)],
            "ask": [1.1001 + i * 1e-5 for i in range(n)],
            "bid_volume": [1.0] * n,
            "ask_volume": [1.0] * n,
        },
        schema={
            "timestamp": pl.Datetime("ms", "UTC"),
            "bid": pl.Float64,
            "ask": pl.Float64,
            "bid_volume": pl.Float64,
            "ask_volume": pl.Float64,
        },
    )
    path = tmp_path / "TEST_2024_DUKASCOPY.parquet"
    df.write_parquet(path)
    return path


@pytest.mark.phase1
def test_load_ticks_returns_lazyframe(tick_parquet: Path) -> None:
    lf = load_ticks(tick_parquet)
    assert isinstance(lf, pl.LazyFrame)


@pytest.mark.phase1
def test_load_ticks_full_collect_yields_all_rows(tick_parquet: Path) -> None:
    df = load_ticks(tick_parquet).collect()
    assert df.height == 1000
    assert {"timestamp", "bid", "ask", "bid_volume", "ask_volume"} <= set(df.columns)


@pytest.mark.phase1
def test_load_ticks_date_filter(tick_parquet: Path) -> None:
    df = load_ticks(
        tick_parquet,
        start=date(2024, 1, 2),  # all data is on 2024-01-01 → empty
    ).collect()
    assert df.height == 0


@pytest.mark.phase1
def test_load_ticks_column_projection(tick_parquet: Path) -> None:
    df = load_ticks(tick_parquet, columns=["timestamp", "bid"]).collect()
    assert set(df.columns) == {"timestamp", "bid"}


@pytest.mark.phase1
def test_load_ticks_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ticks(tmp_path / "does_not_exist.parquet")


@pytest.mark.phase1
def test_verify_schema_passes(tick_parquet: Path) -> None:
    verify_schema(load_ticks(tick_parquet))


@pytest.mark.phase1
def test_verify_schema_detects_missing(tick_parquet: Path) -> None:
    lf = load_ticks(tick_parquet, columns=["timestamp", "bid"])
    with pytest.raises(ValueError, match="missing required columns"):
        verify_schema(lf)
