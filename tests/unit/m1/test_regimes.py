"""M1.3b — configurable holding regimes (first-passage-anchored granularity)."""

from __future__ import annotations

import pytest

from afml.research.regimes import DEFAULT_REGIME, REGIMES, HoldingRegime, get_regime

pytestmark = pytest.mark.m1


def test_day_is_default() -> None:
    assert DEFAULT_REGIME.name == "day"
    assert DEFAULT_REGIME.mean_hold_hours == 6.0
    assert DEFAULT_REGIME.max_hold_hours == 24.0


def test_first_passage_anchor_bar_hours() -> None:
    # Δ = mean_hold / pt²  (pt_reference = 2 ⇒ /4).
    day = get_regime("day")
    assert day.bar_hours == pytest.approx(6.0 / 4.0)  # 1.5h
    # V = round(max_hold / Δ).
    assert day.vertical_bars == round(24.0 / 1.5)  # 16


def test_target_bar_count_scales_with_span() -> None:
    day = get_regime("day")
    # B* = span / Δ.
    assert day.target_bar_count(150.0) == int(150.0 / day.bar_hours)
    assert day.target_bar_count(0.0) == 1  # floored to ≥ 1


def test_regimes_span_scalp_to_position_with_increasing_bar_hours() -> None:
    order = ["scalp", "day", "swing", "position"]
    hours = [REGIMES[name].bar_hours for name in order]
    assert hours == sorted(hours)  # strictly coarser as horizon lengthens
    assert REGIMES["scalp"].bar_hours < REGIMES["day"].bar_hours < REGIMES["swing"].bar_hours


def test_validation() -> None:
    with pytest.raises(ValueError):
        HoldingRegime("bad", mean_hold_hours=0.0, max_hold_hours=1.0)
    with pytest.raises(ValueError):
        HoldingRegime("bad", mean_hold_hours=10.0, max_hold_hours=5.0)  # max < mean
    with pytest.raises(ValueError):
        HoldingRegime("bad", mean_hold_hours=1.0, max_hold_hours=2.0, pt_reference=0.0)


def test_unknown_regime_raises() -> None:
    with pytest.raises(KeyError):
        get_regime("hyperscalp")
