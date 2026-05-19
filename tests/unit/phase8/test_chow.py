"""Phase 8 — Chow-type DF structural-break test (secondary)."""

from __future__ import annotations

import numpy as np
import pytest

from afml.monitoring.chow import chow_break_test


@pytest.mark.phase8
def test_chow_detects_planted_regime_break() -> None:
    """A series whose dynamics flip at the midpoint (random walk → strong
    drift) must trip the Chow F-test."""
    rng = np.random.default_rng(0)
    n = 200
    first = np.cumsum(rng.standard_normal(n // 2)) + 100.0
    # Second half: steep deterministic drift on top of noise.
    second = first[-1] + np.cumsum(rng.standard_normal(n // 2) + 3.0)
    series = np.concatenate([first, second])
    result = chow_break_test(series, breakpoint_frac=0.5)
    assert result.is_break is True
    assert result.f_statistic > result.critical_value


@pytest.mark.phase8
def test_chow_stable_series_no_break() -> None:
    """A homogeneous random walk has no structural break at its midpoint."""
    rng = np.random.default_rng(1)
    series = np.cumsum(rng.standard_normal(200)) + 100.0
    result = chow_break_test(series, breakpoint_frac=0.5)
    assert result.is_break is False


@pytest.mark.phase8
def test_chow_reports_breakpoint_index() -> None:
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.standard_normal(100)) + 100.0
    result = chow_break_test(series, breakpoint_frac=0.4)
    assert result.breakpoint_index == 40


@pytest.mark.phase8
def test_chow_rejects_too_short() -> None:
    with pytest.raises(ValueError, match=r"sub-sample|breakpoint"):
        chow_break_test(np.arange(6, dtype=np.float64), breakpoint_frac=0.5)


@pytest.mark.phase8
def test_chow_rejects_non_finite() -> None:
    bad = np.concatenate([
        np.arange(50, dtype=np.float64),
        [np.nan],
        np.arange(49, dtype=np.float64),
    ])
    with pytest.raises(ValueError, match="finite"):
        chow_break_test(bad)
