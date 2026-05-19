"""Phase 0-8 final audit — Phase 8 patch (V2 GSADF min-window collapse)."""

from __future__ import annotations

import numpy as np
import pytest

from afml.monitoring import StructuralBreakMonitor, detect_bubble
from afml.monitoring.gsadf import (
    MIN_WINDOW_OBSERVATIONS,
    gsadf_statistic,
)


@pytest.mark.phase8
def test_gsadf_short_series_returns_safe_no_bubble() -> None:
    """AFML 0-8 audit V2 — a series shorter than the minimum window must
    return a safe 'no bubble' (statistic 0.0, is_bubble False) instead of
    crashing on an under-determined OLS."""
    rng = np.random.default_rng(0)
    short = np.cumsum(rng.standard_normal(MIN_WINDOW_OBSERVATIONS - 5)) + 100.0
    result = detect_bubble(short, n_simulations=99, random_state=1)
    assert result.is_bubble is False
    assert result.gsadf_statistic == 0.0


@pytest.mark.phase8
def test_gsadf_statistic_short_series_returns_zero() -> None:
    """``gsadf_statistic`` degrades to 0.0 (not an exception) when too short."""
    short = np.linspace(100.0, 101.0, MIN_WINDOW_OBSERVATIONS - 1)
    assert gsadf_statistic(short) == 0.0


@pytest.mark.phase8
def test_gsadf_min_window_floor_enforced() -> None:
    """Even with a tiny ``min_window_frac`` the absolute floor is respected,
    so a borderline-length series still routes through the safe path rather
    than fitting a 2-point regression."""
    rng = np.random.default_rng(0)
    # 10 obs, frac would suggest window=1, but the floor is MIN_WINDOW_OBSERVATIONS.
    series = np.cumsum(rng.standard_normal(10)) + 100.0
    result = detect_bubble(series, min_window_frac=0.01, n_simulations=50, random_state=0)
    assert result.is_bubble is False
    assert result.gsadf_statistic == 0.0


@pytest.mark.phase8
def test_monitor_no_crash_on_short_series() -> None:
    """The Agent 8 monitor must not raise when fed a too-short price window —
    it simply reports no regime break."""
    monitor = StructuralBreakMonitor(n_simulations=50)
    short = np.linspace(100.0, 100.5, 12)
    check = monitor.check_regime("BTCUSD", short, random_state=0)
    assert check.regime_break is False
    assert check.event is None


@pytest.mark.phase8
def test_gsadf_still_detects_bubble_above_floor() -> None:
    """Regression guard — the min-window floor must NOT suppress a genuine
    bubble on a normal-length series."""
    rng = np.random.default_rng(0)
    n = 120
    series = np.cumsum(rng.standard_normal(n)) + 100.0
    series[80:] = series[80] * (1.04 ** np.arange(n - 80))
    result = detect_bubble(series, n_simulations=99, random_state=1)
    assert result.is_bubble is True
