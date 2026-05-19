"""Phase 2 — Primary-alpha plugin registry."""

from __future__ import annotations

import polars as pl
import pytest

from afml.labeling.primary_alphas.base import (
    PrimaryAlpha,
    get_alpha_class,
    list_alpha_families,
    register_alpha,
)


@pytest.mark.phase2
def test_three_families_registered_by_default() -> None:
    families = list_alpha_families()
    assert "cusum" in families
    assert "bbands_meanrev" in families
    assert "donchian_breakout" in families


@pytest.mark.phase2
def test_get_alpha_class_round_trip() -> None:
    for fam in ("cusum", "bbands_meanrev", "donchian_breakout"):
        cls = get_alpha_class(fam)
        assert cls.algorithmic_family == fam


@pytest.mark.phase2
def test_unknown_family_raises() -> None:
    with pytest.raises(KeyError, match="Unknown algorithmic_family"):
        get_alpha_class("never_registered_xyz")


@pytest.mark.phase2
def test_register_requires_family_classvar() -> None:
    with pytest.raises(TypeError, match="algorithmic_family"):

        @register_alpha
        class NoFamily(PrimaryAlpha):
            def detect(self, bars: pl.DataFrame) -> pl.DataFrame:
                return bars


@pytest.mark.phase2
def test_register_is_idempotent_for_same_class() -> None:
    """Re-decorating the same class is a no-op (allows reloads in tests)."""
    cls = get_alpha_class("cusum")
    same = register_alpha(cls)
    assert same is cls


@pytest.mark.phase2
def test_register_rejects_family_collision() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @register_alpha
        class RenamedCUSUM(PrimaryAlpha):
            algorithmic_family = "cusum"

            def detect(self, bars: pl.DataFrame) -> pl.DataFrame:
                return bars


@pytest.mark.phase2
def test_subclass_without_family_raises_on_instantiate() -> None:
    class _Bad(PrimaryAlpha):
        def detect(self, bars: pl.DataFrame) -> pl.DataFrame:
            return bars

    with pytest.raises(TypeError, match="algorithmic_family"):
        _Bad()
