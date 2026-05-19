"""Phase 5 — shared fixtures for the meta-labeller test suite."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass(frozen=True, slots=True)
class MetaLabelDataset:
    """Synthetic Brain-1-style events for Brain-2 testing.

    ``y`` is binary: probability rises with ``X[:, 0]``. ``t0`` / ``t1`` define
    a 5-bar (overlapping) horizon for every event so concurrency varies across
    the bar grid in a realistic way.
    """

    X: np.ndarray
    y: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    n_features: int


@pytest.fixture
def predictable_meta_labels() -> MetaLabelDataset:
    """Brain-1-like events where ``X[:, 0]`` strongly drives the label.

    Used by Phase 5 DoD tests: Brier of calibrated SBRF must beat the
    class-prior baseline on every walk-forward fold.
    """
    rng = np.random.default_rng(20260601)
    n = 1500
    n_features = 6
    X = rng.standard_normal((n, n_features))
    # Latent signal in column 0; the rest are noise.
    latent = X[:, 0] + 0.3 * rng.standard_normal(n)
    y = (latent > 0.0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 5  # 5-bar overlapping horizons
    return MetaLabelDataset(X=X, y=y, t0=t0, t1=t1, n_features=n_features)


@pytest.fixture
def short_overlapping_events() -> MetaLabelDataset:
    """Tiny dataset for fast unit tests of the concurrency / bootstrap layers."""
    rng = np.random.default_rng(0)
    n = 300
    n_features = 3
    X = rng.standard_normal((n, n_features))
    y = (X[:, 0] > 0.0).astype(np.int64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 3
    return MetaLabelDataset(X=X, y=y, t0=t0, t1=t1, n_features=n_features)
