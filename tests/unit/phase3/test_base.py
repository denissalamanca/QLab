"""Phase 3 — FeatureSpec registry mechanics."""

from __future__ import annotations

import pytest

from afml.features.base import (
    FeatureSpec,
    list_features,
    register_feature,
    reset_registry_for_tests,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    reset_registry_for_tests()


@pytest.mark.phase3
def test_register_and_list() -> None:
    spec = FeatureSpec(name="roll_w20", base_family="roll", window=20)
    register_feature(spec)
    assert list_features() == [spec]


@pytest.mark.phase3
def test_register_is_idempotent_for_identical_spec() -> None:
    s = FeatureSpec(name="roll_w20", base_family="roll", window=20)
    register_feature(s)
    register_feature(s)
    assert len(list_features()) == 1


@pytest.mark.phase3
def test_register_rejects_collision_with_different_spec() -> None:
    s1 = FeatureSpec(name="roll_w20", base_family="roll", window=20)
    s2 = FeatureSpec(name="roll_w20", base_family="roll", window=21)
    register_feature(s1)
    with pytest.raises(ValueError, match="already registered"):
        register_feature(s2)


@pytest.mark.phase3
def test_list_features_sorted() -> None:
    for n in ("zeta", "alpha", "mu"):
        register_feature(FeatureSpec(name=n, base_family=n, window=10))
    names = [f.name for f in list_features()]
    assert names == sorted(names)
