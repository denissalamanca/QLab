"""Phase 2 — Bollinger Band Mean-Reversion plugin."""

from __future__ import annotations

import polars as pl
import pytest

from afml.labeling.primary_alphas.bollinger import BollingerMeanReversion


@pytest.mark.phase2
def test_bollinger_returns_schema(bars_mean_reverting: pl.DataFrame) -> None:
    plugin = BollingerMeanReversion(sma_span=20, num_std=2.0)
    events = plugin.detect(bars_mean_reverting)
    assert {"timestamp", "side"} <= set(events.columns)
    if events.height > 0:
        assert set(events["side"].unique().to_list()) <= {"long", "short"}


@pytest.mark.phase2
def test_bollinger_empty_on_short_input() -> None:
    bars = pl.DataFrame(
        {
            "timestamp": [None],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        },
        schema={
            "timestamp": pl.Datetime("ms", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
    )
    plugin = BollingerMeanReversion(sma_span=20, num_std=2.0)
    assert plugin.detect(bars).height == 0


@pytest.mark.phase2
def test_bollinger_higher_num_std_yields_fewer_events(
    bars_mean_reverting: pl.DataFrame,
) -> None:
    narrow = BollingerMeanReversion(sma_span=20, num_std=1.0).detect(bars_mean_reverting)
    wide = BollingerMeanReversion(sma_span=20, num_std=3.0).detect(bars_mean_reverting)
    assert wide.height <= narrow.height


@pytest.mark.phase2
def test_bollinger_fires_on_mean_reverting_data(
    bars_mean_reverting: pl.DataFrame,
) -> None:
    """On an OU-like series there must be at least *some* band crossings."""
    plugin = BollingerMeanReversion(sma_span=20, num_std=1.5)
    events = plugin.detect(bars_mean_reverting)
    assert events.height > 0


@pytest.mark.phase2
def test_bollinger_params_round_trip() -> None:
    plugin = BollingerMeanReversion(sma_span=30, num_std=2.5)
    assert plugin.params == {"sma_span": 30, "num_std": 2.5}
    assert plugin.algorithmic_family == "bbands_meanrev"
