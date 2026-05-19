"""The 14-asset universe — canonical registry.

Single source of truth for which instruments the AFML pipeline trades. Per-asset
metadata (class, file pattern, available data range, pip size, market hours) lives
here. Changing the universe is a deliberate, reviewable code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path


class AssetClass(StrEnum):
    FX = "fx"
    INDEX = "index"
    METAL = "metal"
    CRYPTO = "crypto"


@dataclass(frozen=True, slots=True)
class AssetSpec:
    """Immutable description of one tradeable instrument."""

    symbol: str
    asset_class: AssetClass
    data_filename: str
    data_start: date
    data_end: date
    pip_size: float
    is_24_7: bool

    @property
    def data_path(self) -> Path:
        return DATA_ROOT / self.data_filename


DATA_ROOT: Path = Path(
    "/Users/dsalamanca/vs_env/Antigravity/Quant Lab/data/multi_year_consolidated"
)


# ---------------------------------------------------------------------------------
# The 14-asset universe (locked per PRD + clarification).
# ---------------------------------------------------------------------------------
ASSETS: tuple[AssetSpec, ...] = (
    # FX (8)
    AssetSpec(
        "EURUSD",
        AssetClass.FX,
        "EURUSD_2016_2025_DUKASCOPY.parquet",
        date(2016, 1, 1),
        date(2025, 12, 31),
        0.0001,
        False,
    ),
    AssetSpec(
        "GBPUSD",
        AssetClass.FX,
        "GBPUSD_2016_2025_DUKASCOPY.parquet",
        date(2016, 1, 1),
        date(2025, 12, 31),
        0.0001,
        False,
    ),
    AssetSpec(
        "USDJPY",
        AssetClass.FX,
        "USDJPY_2020_2025_DUKASCOPY.parquet",
        date(2020, 1, 1),
        date(2025, 12, 31),
        0.01,
        False,
    ),
    AssetSpec(
        "AUDUSD",
        AssetClass.FX,
        "AUDUSD_2020_2025_DUKASCOPY.parquet",
        date(2020, 1, 1),
        date(2025, 12, 31),
        0.0001,
        False,
    ),
    AssetSpec(
        "USDCHF",
        AssetClass.FX,
        "USDCHF_2020_2025_DUKASCOPY.parquet",
        date(2020, 1, 1),
        date(2025, 12, 31),
        0.0001,
        False,
    ),
    AssetSpec(
        "NZDUSD",
        AssetClass.FX,
        "NZDUSD_2020_2025_DUKASCOPY.parquet",
        date(2020, 1, 1),
        date(2025, 12, 31),
        0.0001,
        False,
    ),
    AssetSpec(
        "EURGBP",
        AssetClass.FX,
        "EURGBP_2020_2025_DUKASCOPY.parquet",
        date(2020, 1, 1),
        date(2025, 12, 31),
        0.0001,
        False,
    ),
    AssetSpec(
        "EURJPY",
        AssetClass.FX,
        "EURJPY_2020_2025_DUKASCOPY.parquet",
        date(2020, 1, 1),
        date(2025, 12, 31),
        0.01,
        False,
    ),
    # Indices (2)
    AssetSpec(
        "DAX",
        AssetClass.INDEX,
        "DAX_2020_2025_DUKASCOPY.parquet",
        date(2020, 1, 1),
        date(2025, 12, 31),
        0.01,
        False,
    ),
    AssetSpec(
        "USA500",
        AssetClass.INDEX,
        "USA500_2020_2025_DUKASCOPY.parquet",
        date(2020, 1, 1),
        date(2025, 12, 31),
        0.01,
        False,
    ),
    # Metals (2)
    AssetSpec(
        "XAUUSD",
        AssetClass.METAL,
        "XAUUSD_2020_2025_DUKASCOPY.parquet",
        date(2020, 1, 1),
        date(2025, 12, 31),
        0.01,
        False,
    ),
    AssetSpec(
        "XAGUSD",
        AssetClass.METAL,
        "XAGUSD_2020_2025_DUKASCOPY.parquet",
        date(2020, 1, 1),
        date(2025, 12, 31),
        0.001,
        False,
    ),
    # Crypto (2) — 24/7 markets
    AssetSpec(
        "BTCUSD",
        AssetClass.CRYPTO,
        "BTCUSD_2020_2025_DUKASCOPY.parquet",
        date(2020, 1, 1),
        date(2025, 12, 31),
        0.01,
        True,
    ),
    AssetSpec(
        "ETHUSD",
        AssetClass.CRYPTO,
        "ETHUSD_2020_2025_DUKASCOPY.parquet",
        date(2020, 1, 1),
        date(2025, 12, 31),
        0.01,
        True,
    ),
)


_ASSETS_BY_SYMBOL: dict[str, AssetSpec] = {a.symbol: a for a in ASSETS}


def get_asset(symbol: str) -> AssetSpec:
    """Look up an asset by symbol. Raises KeyError if outside the universe."""
    try:
        return _ASSETS_BY_SYMBOL[symbol]
    except KeyError as e:
        raise KeyError(f"Unknown asset {symbol!r}; universe = {list(_ASSETS_BY_SYMBOL)}") from e


def assert_universe_complete() -> None:
    """Verify the 14-asset universe has all data files on disk.

    Used at agent boot. Raises FileNotFoundError if any asset's parquet is missing.
    """
    expected = 14
    if len(ASSETS) != expected:
        raise AssertionError(f"Universe must have exactly {expected} assets, got {len(ASSETS)}")
    for asset in ASSETS:
        if not asset.data_path.exists():
            raise FileNotFoundError(f"Missing data file for {asset.symbol}: {asset.data_path}")
