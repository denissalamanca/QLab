"""Phase 4 shared fixtures — synthetic signal-vs-noise feature matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """A bundled signal-vs-noise feature matrix + binary label for Phase 4 tests.

    The structure is deliberate:

    - ``signal_cols``: features that carry the latent classifier signal (one
      tight cluster).
    - ``correlated_noise_cols``: pairs / triples of features that are mutually
      correlated but independent of ``y`` (multiple noise clusters).
    - ``independent_noise_cols``: pure white noise (one cluster per).
    """

    X: pl.DataFrame
    y: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    signal_cols: list[str]
    correlated_noise_cols: list[str]
    independent_noise_cols: list[str]
    feature_columns: list[str]


@pytest.fixture
def synthetic_classification() -> SyntheticDataset:
    """Synthetic dataset where 3 features carry signal, 9 are noise.

    Designed so:
    - ONC must find ≥ 4 clusters (1 signal + ≥ 3 noise structures).
    - Clustered MDA must keep the signal cluster (p < 0.05, importance ≫ 0).
    - Clustered MDA must drop the noise clusters (importance ≈ 0, p > 0.05).
    - Dimensionality reduces from 12 → 3 (75 % reduction; > 20 % gate).
    """
    rng = np.random.default_rng(20260519)
    n = 800
    latent = rng.standard_normal(n)
    y = (latent > 0.0).astype(np.int64)

    cols: dict[str, np.ndarray] = {}

    # Signal cluster: 3 noisy copies of the latent variable.
    signal_cols = [f"sig_{i}" for i in range(3)]
    for c in signal_cols:
        cols[c] = latent + rng.standard_normal(n) * 0.25

    # Correlated noise cluster A: 3 features sharing a hidden factor.
    factor_a = rng.standard_normal(n)
    corr_a_cols = [f"corrA_{i}" for i in range(3)]
    for c in corr_a_cols:
        cols[c] = factor_a + rng.standard_normal(n) * 0.30

    # Correlated noise cluster B: 3 features sharing a different factor.
    factor_b = rng.standard_normal(n)
    corr_b_cols = [f"corrB_{i}" for i in range(3)]
    for c in corr_b_cols:
        cols[c] = factor_b + rng.standard_normal(n) * 0.30

    correlated_noise_cols = corr_a_cols + corr_b_cols

    # Independent noise: 3 pure-noise features.
    indep_cols = [f"indep_{i}" for i in range(3)]
    for c in indep_cols:
        cols[c] = rng.standard_normal(n)

    feature_columns = signal_cols + corr_a_cols + corr_b_cols + indep_cols
    X = pl.DataFrame({c: cols[c] for c in feature_columns})

    # t0 / t1 — non-overlapping unit horizons.
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 5  # 5-bar horizon; consecutive samples WILL overlap → exercises purging.

    return SyntheticDataset(
        X=X,
        y=y,
        t0=t0,
        t1=t1,
        signal_cols=signal_cols,
        correlated_noise_cols=correlated_noise_cols,
        independent_noise_cols=indep_cols,
        feature_columns=feature_columns,
    )


@pytest.fixture
def synthetic_all_noise() -> SyntheticDataset:
    """All-noise dataset — no feature predicts ``y``.

    On this fixture Clustered MDA correctly drops every cluster (none are
    statistically significant against random labels). Survives the
    ``select_features`` ≥ 20 % reduction gate trivially — *because reduction
    is 100 %* — so the SFI fallback does NOT trigger. Used by
    ``test_pipeline_all_noise_drops_everything``.
    """
    rng = np.random.default_rng(7777)
    n = 600
    n_features = 8
    cols = {f"noise_{i}": rng.standard_normal(n) for i in range(n_features)}
    X = pl.DataFrame(cols)
    y = rng.integers(0, 2, size=n)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 3
    feature_columns = list(cols.keys())
    return SyntheticDataset(
        X=X,
        y=y,
        t0=t0,
        t1=t1,
        signal_cols=[],
        correlated_noise_cols=[],
        independent_noise_cols=feature_columns,
        feature_columns=feature_columns,
    )


@pytest.fixture
def synthetic_redundant_signal() -> SyntheticDataset:
    """Every feature is a noisy copy of the same latent signal.

    Clustered MDA sees one big informative cluster carrying every feature; it
    keeps the cluster, leaving *all* features alive ⇒ 0 % reduction ⇒ the
    pipeline triggers the SFI fallback, which then ranks features individually
    and prunes the weakest ones to enforce the ≥ 20 % reduction floor.

    Used by ``test_pipeline_triggers_sfi_fallback_when_mda_keeps_everything``.
    """
    rng = np.random.default_rng(20260520)
    n = 800
    latent = rng.standard_normal(n)
    y = (latent > 0.0).astype(np.int64)
    n_features = 10
    cols = {
        f"redundant_{i}": latent + rng.standard_normal(n) * (0.25 + 0.05 * i)
        for i in range(n_features)
    }
    X = pl.DataFrame(cols)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    feature_columns = list(cols.keys())
    return SyntheticDataset(
        X=X,
        y=y,
        t0=t0,
        t1=t1,
        signal_cols=feature_columns,
        correlated_noise_cols=[],
        independent_noise_cols=[],
        feature_columns=feature_columns,
    )
