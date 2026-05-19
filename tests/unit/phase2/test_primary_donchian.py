"""Phase 2 — Donchian Channel Breakout plugin."""

from __future__ import annotations

import polars as pl
import pytest

from afml.labeling.primary_alphas.donchian import DonchianBreakout


@pytest.mark.phase2
def test_donchian_returns_schema(bars_trending: pl.DataFrame) -> None:
    plugin = DonchianBreakout(window=20)
    events = plugin.detect(bars_trending)
    assert {"timestamp", "side"} <= set(events.columns)
    if events.height > 0:
        assert set(events["side"].unique().to_list()) <= {"long", "short"}


@pytest.mark.phase2
def test_donchian_fires_on_trending_series(bars_trending: pl.DataFrame) -> None:
    plugin = DonchianBreakout(window=20)
    events = plugin.detect(bars_trending)
    assert events.height > 0


@pytest.mark.phase2
def test_donchian_smaller_window_more_events(bars_trending: pl.DataFrame) -> None:
    small = DonchianBreakout(window=10).detect(bars_trending)
    large = DonchianBreakout(window=80).detect(bars_trending)
    assert small.height >= large.height


@pytest.mark.phase2
def test_donchian_empty_on_constant_prices(bars_constant: pl.DataFrame) -> None:
    plugin = DonchianBreakout(window=20)
    events = plugin.detect(bars_constant)
    assert events.height == 0


@pytest.mark.phase2
def test_donchian_does_not_double_fire_inside_breakout(
    bars_trending: pl.DataFrame,
) -> None:
    """The breakout flag must reset only when price returns inside the channel,
    so consecutive bars all outside the channel produce ONE event, not many.

    Operationally: total events << total bars when the series is trending.
    """
    plugin = DonchianBreakout(window=20)
    events = plugin.detect(bars_trending)
    # Far fewer events than bars: even with a noisy trend we don't fire on
    # every bar — the breakout flag must reset only when price re-enters
    # the channel.
    assert events.height < bars_trending.height // 5


@pytest.mark.phase2
def test_donchian_params_round_trip() -> None:
    plugin = DonchianBreakout(window=15)
    assert plugin.params == {"window": 15}
    assert plugin.algorithmic_family == "donchian_breakout"
