"""Phase 6 — FWER controls (Bonferroni + Holm)."""

from __future__ import annotations

import numpy as np
import pytest

from afml.validation.fwer import bonferroni_threshold, holm_bonferroni


@pytest.mark.phase6
def test_bonferroni_threshold_basic() -> None:
    assert bonferroni_threshold(0.05, 20) == pytest.approx(0.0025)
    assert bonferroni_threshold(0.01, 100) == pytest.approx(0.0001)


@pytest.mark.phase6
def test_bonferroni_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="alpha"):
        bonferroni_threshold(0.0, 10)
    with pytest.raises(ValueError, match="alpha"):
        bonferroni_threshold(1.0, 10)
    with pytest.raises(ValueError, match="n_trials"):
        bonferroni_threshold(0.05, 0)


@pytest.mark.phase6
def test_holm_bonferroni_rejects_all_small_pvalues() -> None:
    """If every p-value is tiny, Holm should reject every null."""
    p = np.full(10, 1e-6)
    mask = holm_bonferroni(p, alpha=0.05)
    assert mask.all()


@pytest.mark.phase6
def test_holm_bonferroni_rejects_none_when_all_p_one() -> None:
    p = np.ones(10)
    mask = holm_bonferroni(p, alpha=0.05)
    assert not mask.any()


@pytest.mark.phase6
def test_holm_bonferroni_more_powerful_than_bonferroni() -> None:
    """Holm strictly dominates Bonferroni: every Bonferroni rejection is
    also a Holm rejection."""
    rng = np.random.default_rng(0)
    p = rng.beta(0.5, 5.0, size=20)  # heavy-mass-near-zero distribution
    alpha = 0.05
    bonf_mask = p < bonferroni_threshold(alpha, p.size)
    holm_mask = holm_bonferroni(p, alpha)
    # Holm rejection set ⊇ Bonferroni rejection set.
    assert np.all(holm_mask[bonf_mask])


@pytest.mark.phase6
def test_holm_bonferroni_step_down_behavior() -> None:
    """First failure stops the rejection cascade."""
    p = np.array([0.001, 0.002, 0.5, 0.0005, 0.6])
    mask = holm_bonferroni(p, alpha=0.05)
    # Sorted: 0.0005, 0.001, 0.002, 0.5, 0.6
    # Thresholds: 0.05/5=0.01, 0.05/4=0.0125, 0.05/3=0.01667, 0.05/2=0.025, 0.05/1=0.05
    # First three p-values are below their thresholds; the fourth (0.5) fails;
    # so we reject only the first three (in sorted order).
    assert mask.sum() == 3
