"""Phase 1 DoD — the Truncation Hash Test.

Blueprint §3.3:

    hash_full = hashlib.sha256(pd.util.hash_pandas_object(ffd_full.iloc[:5000])).hexdigest()
    hash_trunc = hashlib.sha256(pd.util.hash_pandas_object(ffd_trunc.iloc[:5000])).hexdigest()
    assert hash_full == hash_trunc

We re-state it in numpy with ``afml.data.causality.truncation_hash`` and
``assert_no_leakage``. The fixed-width FFD must produce byte-identical output
over the overlap of a full and a truncated computation — proving no future
data leaks into past rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from afml.data.causality import assert_no_leakage, truncation_hash
from afml.data.ffd import ffd_apply


@pytest.mark.phase1
def test_ffd_truncation_hash_invariant() -> None:
    """``ffd_apply`` is causal: full output prefix equals truncated output exactly.

    After AFML audit V2, ``ffd_apply`` returns length ``n - l*``. Output index 0
    of both full and truncated runs corresponds to the same underlying input
    convolution — they must be byte-identical for the first ``len(trunc_ffd)``
    rows.
    """
    rng = np.random.default_rng(20260519)
    full_series = np.cumsum(rng.standard_normal(4000))
    truncation_point = 2500
    truncated_series = full_series[:truncation_point]

    d = 0.4
    full_ffd = ffd_apply(full_series, d=d)
    trunc_ffd = ffd_apply(truncated_series, d=d)

    overlap_end = len(trunc_ffd)
    h_full = truncation_hash(full_ffd[:overlap_end])
    h_trunc = truncation_hash(trunc_ffd[:overlap_end])
    assert h_full == h_trunc, (
        f"Future-data leakage detected: full {h_full[:12]}… != truncated {h_trunc[:12]}…"
    )
    # Element-wise check too, for an actionable error if the hash test ever fails.
    assert_no_leakage(full_ffd, trunc_ffd, overlap_start=0, overlap_end=overlap_end)


@pytest.mark.phase1
def test_ffd_truncation_hash_invariant_multiple_d() -> None:
    """The causality guarantee holds across the entire ``d ∈ (0, 1]`` range."""
    rng = np.random.default_rng(42)
    full = np.cumsum(rng.standard_normal(3000))
    trunc_point = 1500

    for d in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        full_ffd = ffd_apply(full, d=d)
        trunc_ffd = ffd_apply(full[:trunc_point], d=d)
        if trunc_ffd.size == 0:
            # Truncation point smaller than the window — no overlap to verify.
            continue
        assert_no_leakage(
            full_ffd,
            trunc_ffd,
            overlap_start=0,
            overlap_end=trunc_ffd.size,
        )


@pytest.mark.phase1
def test_truncation_hash_detects_a_synthetic_leak() -> None:
    """If we synthesize a leaky transform that uses GLOBAL statistics (i.e. future
    data via aggregation), the hash test must fail. This guards the test
    infrastructure: we want certainty it would catch a real causality bug.
    """
    rng = np.random.default_rng(0)
    series = rng.standard_normal(500)

    # A global z-score is leaky: each output uses mean(WHOLE series) and
    # std(WHOLE series), so truncating the input changes every output value.
    def leaky_global_zscore(s: np.ndarray) -> np.ndarray:
        return (s - float(np.mean(s))) / float(np.std(s))

    full_out = leaky_global_zscore(series)
    trunc_out = leaky_global_zscore(series[:250])
    with pytest.raises(AssertionError):
        assert_no_leakage(full_out, trunc_out, overlap_start=0, overlap_end=250)
