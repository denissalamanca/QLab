"""``PrimaryAlpha`` plugin interface + registry.

Every Brain 1 algorithmic family is a subclass of ``PrimaryAlpha`` decorated
with ``@register_alpha``. The registry exposes them by ``algorithmic_family``
key — the same string written to the Alpha Registry's
``algorithmic_family`` column.

Why a plugin interface: PRD §5 Dual-Path Event Sampling requires "native primary
alpha entry logic" in addition to CUSUM. Agent 2 parameter-sweeps each plugin
across its own hyperparameter grid and logs every resulting hypothesis to the
Alpha Registry, so the multiple-testing penalty (DSR / FWER, Phase 6) is honest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import polars as pl

PrimaryAlphaParams = dict[str, Any]


class PrimaryAlpha(ABC):
    """Abstract base for Brain 1 primary-alpha plugins.

    Subclasses must:
    1. Declare ``algorithmic_family`` as a ``ClassVar[str]`` (lowercase, snake_case).
    2. Accept all tunable hyperparameters as keyword arguments in ``__init__``,
       storing them on ``self.params`` so they round-trip to the Alpha Registry's
       ``hyperparameter_vector`` column verbatim.
    3. Implement ``detect(bars)`` returning a Polars DataFrame with at minimum:
       ``timestamp`` (event time, equal to a bar close), ``side`` (Literal
       ``"long"`` or ``"short"``). Strictly chronological.
    """

    algorithmic_family: ClassVar[str] = ""

    def __init__(self, **params: Any) -> None:
        if not self.algorithmic_family:
            raise TypeError(f"{type(self).__name__} must set ``algorithmic_family`` ClassVar")
        self.params: PrimaryAlphaParams = dict(params)

    @abstractmethod
    def detect(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Detect events on a bar sequence.

        Parameters
        ----------
        bars : Polars DataFrame with at minimum ``timestamp``, ``open``, ``high``,
            ``low``, ``close``. Sorted by ``timestamp`` ascending.

        Returns
        -------
        DataFrame with columns ``timestamp`` (= bar close at event time) and
        ``side`` (``"long"`` / ``"short"``), sorted ascending.
        """
        ...


# ---------------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------------
_REGISTRY: dict[str, type[PrimaryAlpha]] = {}


def register_alpha(cls: type[PrimaryAlpha]) -> type[PrimaryAlpha]:
    """Class decorator: register ``cls`` under its ``algorithmic_family``."""
    if not cls.algorithmic_family:
        raise TypeError(f"@register_alpha applied to {cls.__name__} which lacks algorithmic_family")
    if cls.algorithmic_family in _REGISTRY:
        existing = _REGISTRY[cls.algorithmic_family]
        if existing is not cls:
            raise ValueError(
                f"algorithmic_family {cls.algorithmic_family!r} already registered "
                f"by {existing.__name__}"
            )
    _REGISTRY[cls.algorithmic_family] = cls
    return cls


def get_alpha_class(family: str) -> type[PrimaryAlpha]:
    """Look up a plugin class by ``algorithmic_family``."""
    try:
        return _REGISTRY[family]
    except KeyError as e:
        raise KeyError(f"Unknown algorithmic_family {family!r}; known: {sorted(_REGISTRY)}") from e


def list_alpha_families() -> list[str]:
    """All registered ``algorithmic_family`` keys (sorted)."""
    return sorted(_REGISTRY)


def _reset_registry_for_tests() -> None:
    """Internal helper — clears the registry. Tests only."""
    _REGISTRY.clear()
