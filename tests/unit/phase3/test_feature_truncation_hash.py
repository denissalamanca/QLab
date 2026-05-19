"""Phase 3 DoD — Per-feature Truncation Hash Test.

Blueprint §5.3:

    "Causality Hash Test: feature vectors calculated on truncated datasets must
     perfectly match full-dataset calculations."

For each microstructure feature we compute it on the full bar series and on a
truncated prefix, then assert byte-equivalent output over the overlap. This
mathematically proves zero future-data leakage in every rolling computation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from afml.data.causality import assert_no_leakage
from afml.features.amihud import amihud_lambda
from afml.features.corwin_schultz import corwin_schultz_spread
from afml.features.hasbrouck import hasbrouck_flow
from afml.features.kyle import kyle_lambda
from afml.features.lempel_ziv import lempel_ziv_complexity
from afml.features.ofi import ofi
from afml.features.roll import roll_measure
from afml.features.shannon import shannon_entropy

TRUNCATION_POINT = 1300
WINDOW = 50

BarTuple = tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]


@pytest.fixture(scope="module")
def synthetic_bars() -> BarTuple:
    rng = np.random.default_rng(2024_05_19)
    n = 2000
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
    high = close + np.abs(rng.normal(0.01, 0.005, n))
    low = close - np.abs(rng.normal(0.01, 0.005, n))
    volume = rng.uniform(1.0, 10.0, n)
    return close, high, low, volume


def _assert_feature_causal(
    feature_full: npt.NDArray[np.float64],
    feature_trunc: npt.NDArray[np.float64],
    *,
    name: str,
) -> None:
    """Compare the full and truncated outputs over the overlap region."""
    # The valid overlap starts at WINDOW (warm-up) and ends at the truncation
    # point (truncated array's length).
    assert_no_leakage(
        feature_full,
        feature_trunc,
        overlap_start=WINDOW,
        overlap_end=TRUNCATION_POINT,
    )


@pytest.mark.phase3
def test_roll_truncation_hash(synthetic_bars: BarTuple) -> None:
    close, _, _, _ = synthetic_bars
    full = roll_measure(close, window=WINDOW)
    trunc = roll_measure(close[:TRUNCATION_POINT], window=WINDOW)
    _assert_feature_causal(full, trunc, name="roll")


@pytest.mark.phase3
def test_corwin_schultz_truncation_hash(synthetic_bars: BarTuple) -> None:
    _, high, low, _ = synthetic_bars
    full = corwin_schultz_spread(high, low, window=2)
    trunc = corwin_schultz_spread(high[:TRUNCATION_POINT], low[:TRUNCATION_POINT], window=2)
    assert_no_leakage(full, trunc, overlap_start=10, overlap_end=TRUNCATION_POINT)


@pytest.mark.phase3
def test_ofi_truncation_hash(synthetic_bars: BarTuple) -> None:
    close, _, _, volume = synthetic_bars
    full = ofi(close, volume, window=WINDOW)
    trunc = ofi(close[:TRUNCATION_POINT], volume[:TRUNCATION_POINT], window=WINDOW)
    _assert_feature_causal(full, trunc, name="ofi")


@pytest.mark.phase3
def test_kyle_truncation_hash(synthetic_bars: BarTuple) -> None:
    close, _, _, volume = synthetic_bars
    full = kyle_lambda(close, volume, window=WINDOW)
    trunc = kyle_lambda(close[:TRUNCATION_POINT], volume[:TRUNCATION_POINT], window=WINDOW)
    _assert_feature_causal(full, trunc, name="kyle")


@pytest.mark.phase3
def test_amihud_truncation_hash(synthetic_bars: BarTuple) -> None:
    close, _, _, volume = synthetic_bars
    full = amihud_lambda(close, volume, window=WINDOW)
    trunc = amihud_lambda(close[:TRUNCATION_POINT], volume[:TRUNCATION_POINT], window=WINDOW)
    _assert_feature_causal(full, trunc, name="amihud")


@pytest.mark.phase3
def test_hasbrouck_truncation_hash(synthetic_bars: BarTuple) -> None:
    close, _, _, volume = synthetic_bars
    full = hasbrouck_flow(close, volume, window=WINDOW)
    trunc = hasbrouck_flow(close[:TRUNCATION_POINT], volume[:TRUNCATION_POINT], window=WINDOW)
    _assert_feature_causal(full, trunc, name="hasbrouck")


@pytest.mark.phase3
def test_shannon_truncation_hash(synthetic_bars: BarTuple) -> None:
    close, _, _, _ = synthetic_bars
    full = shannon_entropy(close, window=WINDOW)
    trunc = shannon_entropy(close[:TRUNCATION_POINT], window=WINDOW)
    _assert_feature_causal(full, trunc, name="shannon")


@pytest.mark.phase3
def test_lempel_ziv_truncation_hash(synthetic_bars: BarTuple) -> None:
    close, _, _, _ = synthetic_bars
    full = lempel_ziv_complexity(close, window=WINDOW)
    trunc = lempel_ziv_complexity(close[:TRUNCATION_POINT], window=WINDOW)
    _assert_feature_causal(full, trunc, name="lempel_ziv")


# Parametrized form for a single source of truth — if a new feature is added
# its truncation invariant is enforced just by appending to this tuple.
@pytest.mark.phase3
@pytest.mark.parametrize(
    ("name", "func", "uses_volume", "uses_high_low"),
    [
        ("roll", roll_measure, False, False),
        ("ofi", ofi, True, False),
        ("kyle", kyle_lambda, True, False),
        ("amihud", amihud_lambda, True, False),
        ("hasbrouck", hasbrouck_flow, True, False),
        ("shannon", shannon_entropy, False, False),
        ("lempel_ziv", lempel_ziv_complexity, False, False),
    ],
)
def test_all_features_causal_under_truncation(
    synthetic_bars: BarTuple,
    name: str,
    func: Callable[..., npt.NDArray[np.float64]],
    uses_volume: bool,
    uses_high_low: bool,
) -> None:
    """Single parametrized guard: every registered base feature must satisfy
    the truncation-hash invariant. Adding a new feature => add a row here."""
    close, _, _, volume = synthetic_bars
    if uses_volume:
        full = func(close, volume, window=WINDOW)
        trunc = func(close[:TRUNCATION_POINT], volume[:TRUNCATION_POINT], window=WINDOW)
    else:
        full = func(close, window=WINDOW)
        trunc = func(close[:TRUNCATION_POINT], window=WINDOW)
    _assert_feature_causal(full, trunc, name=name)
