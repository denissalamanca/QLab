"""Research layer (Ops M1) — the historical sweep that turns the validated AFML
engine loose on real data to populate the Alpha Registry with certified trials.

Public surface grows milestone by milestone; M1.1 ships the plateau selector.
"""

from afml.research.grids import (
    FAMILY_GRIDS,
    FamilyGrid,
    GridAxis,
    get_family_grid,
)
from afml.research.harness import (
    EventDataset,
    TrialResult,
    build_event_dataset,
    run_trial,
)
from afml.research.objective import oos_strategy_sharpe
from afml.research.plateau import Coord, PlateauResult, select_plateau
from afml.research.precompute import AssetPrecompute, precompute_asset
from afml.research.regimes import DEFAULT_REGIME, REGIMES, HoldingRegime, get_regime
from afml.research.sweep import (
    CertificationResult,
    SweepCertification,
    SweepResult,
    certify,
    run_sweep,
    sweep_and_certify,
)

__all__ = [
    "DEFAULT_REGIME",
    "FAMILY_GRIDS",
    "REGIMES",
    "AssetPrecompute",
    "CertificationResult",
    "Coord",
    "EventDataset",
    "FamilyGrid",
    "GridAxis",
    "HoldingRegime",
    "PlateauResult",
    "SweepCertification",
    "SweepResult",
    "TrialResult",
    "build_event_dataset",
    "certify",
    "get_family_grid",
    "get_regime",
    "oos_strategy_sharpe",
    "precompute_asset",
    "run_sweep",
    "run_trial",
    "select_plateau",
    "sweep_and_certify",
]
