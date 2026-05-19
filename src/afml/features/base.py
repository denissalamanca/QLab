"""``FeatureSpec`` + global feature registry (Phase 3 substrate).

Every microstructure feature emitted by the pipeline carries a ``FeatureSpec``
describing its base family, window, and the canonical column name. The registry
is queried by Phase 4 ONC (to know which features are clusterable) and by the
Alpha Registry (to log the feature set tied to each experiment).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Metadata for one column in the Phase 3 feature matrix."""

    name: str  # canonical column name, e.g. "roll_w20"
    base_family: str  # "roll" | "corwin_schultz" | "ofi" | "kyle" | ...
    window: int
    causal: bool = True
    description: str = ""


_FEATURE_REGISTRY: dict[str, FeatureSpec] = {}


def register_feature(spec: FeatureSpec) -> FeatureSpec:
    """Add ``spec`` to the global registry. Idempotent if the spec is identical."""
    existing = _FEATURE_REGISTRY.get(spec.name)
    if existing is not None and existing != spec:
        raise ValueError(
            f"Feature name {spec.name!r} already registered with a different spec "
            f"(existing={existing!r}, new={spec!r})"
        )
    _FEATURE_REGISTRY[spec.name] = spec
    return spec


def list_features() -> list[FeatureSpec]:
    """All registered feature specs sorted by name."""
    return sorted(_FEATURE_REGISTRY.values(), key=lambda f: f.name)


def reset_registry_for_tests() -> None:
    """Empty the registry. Tests only — used to keep cross-test state isolated."""
    _FEATURE_REGISTRY.clear()
