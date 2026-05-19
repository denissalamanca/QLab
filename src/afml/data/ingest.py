"""Dukascopy parquet tick loader (Polars, lazy / out-of-core).

The raw tick files are 1–4 GB each. ``load_ticks`` returns a
``polars.LazyFrame`` so the caller can compose filters and projections that get
pushed down to the parquet reader — never materializing the full frame unless
explicitly requested.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from pathlib import Path

import polars as pl

from afml.config.assets import AssetSpec

REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ask",
    "bid",
    "ask_volume",
    "bid_volume",
)


def load_ticks(
    asset: AssetSpec | Path | str,
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    columns: list[str] | None = None,
) -> pl.LazyFrame:
    """Open the Dukascopy parquet for ``asset`` as a lazy frame.

    Parameters
    ----------
    asset : either an ``AssetSpec`` from the universe registry, or a direct path
        to a parquet file.
    start, end : optional half-open date/datetime range filter (``[start, end)``).
        Naïve dates are interpreted as 00:00 UTC.
    columns : optional projection. Defaults to all columns.

    Returns
    -------
    LazyFrame with at minimum the AFML standard schema:
    ``timestamp[ms,UTC], ask, bid, ask_volume, bid_volume``.
    """
    path = asset.data_path if isinstance(asset, AssetSpec) else Path(asset)
    if not path.exists():
        raise FileNotFoundError(f"Tick file not found: {path}")

    lf = pl.scan_parquet(str(path))

    if columns is not None:
        lf = lf.select(columns)

    if start is not None:
        lf = lf.filter(pl.col("timestamp") >= _to_utc_datetime(start))
    if end is not None:
        lf = lf.filter(pl.col("timestamp") < _to_utc_datetime(end))

    return lf


def _to_utc_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    return datetime.combine(value, time.min, tzinfo=UTC)


def verify_schema(lf: pl.LazyFrame) -> None:
    """Assert the lazy frame exposes the AFML-standard tick columns.

    Cheap (only fetches the schema, no data). Run once at agent boot.
    """
    schema = lf.collect_schema()
    missing = [c for c in REQUIRED_COLUMNS if c not in schema.names()]
    if missing:
        raise ValueError(f"Tick frame missing required columns {missing}; found {schema.names()}")
