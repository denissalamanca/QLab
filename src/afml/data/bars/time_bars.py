"""Time bars — fixed-interval OHLC aggregation.

The simplest (and informationally weakest) bar type. Included as a baseline in
the Jarque-Bera tournament: if time bars yield more-normal returns than the
information-driven alternatives, that's the chosen sampler.
"""

from __future__ import annotations

import polars as pl


def build_time_bars(ticks: pl.LazyFrame | pl.DataFrame, interval: str = "1m") -> pl.DataFrame:
    """Aggregate raw ticks into fixed-interval OHLC bars.

    Parameters
    ----------
    ticks : Polars frame with columns ``timestamp`` (datetime, UTC), ``bid``,
        ``ask``, ``bid_volume``, ``ask_volume``.
    interval : Polars duration string — e.g. ``"1m"``, ``"5m"``, ``"1h"``.

    Returns
    -------
    DataFrame with columns ``timestamp`` (bar open), ``open``, ``high``, ``low``,
    ``close``, ``volume``, ``n_ticks``. Mid-price is used for OHLC.
    """
    lf = ticks if isinstance(ticks, pl.LazyFrame) else ticks.lazy()
    return (
        lf
        .with_columns(((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"))
        .group_by_dynamic("timestamp", every=interval)
        .agg(
            pl.col("mid").first().alias("open"),
            pl.col("mid").max().alias("high"),
            pl.col("mid").min().alias("low"),
            pl.col("mid").last().alias("close"),
            (pl.col("bid_volume") + pl.col("ask_volume")).sum().alias("volume"),
            pl.col("mid").count().alias("n_ticks"),
        )
        .filter(pl.col("n_ticks") > 0)
        .sort("timestamp")
        .collect()
    )
