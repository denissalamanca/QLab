"""SQLAlchemy 2.0 ORM schema for the Historical Alpha Registry.

The Alpha Registry is the cryptographic truth-of-record for every hypothesis the
lab has tested. Its trial count directly drives the Deflated Sharpe Ratio penalty
in Phase 6.

Schema follows Master Engineering Blueprint §2.2.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


metadata = Base.metadata


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Experiment(Base):
    """One hypothesis tested against one asset under one parameter vector."""

    __tablename__ = "experiments"

    experiment_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    agent_version: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
        index=True,
    )
    asset: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    algorithmic_family: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    hyperparameter_vector: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    hyperparameter_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    orthogonality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_events_triggered: Mapped[int] = mapped_column(Integer, nullable=False)
    brain_1_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    brain_2_log_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_deployed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "asset",
            "algorithmic_family",
            "hyperparameter_hash",
            name="uq_experiment_dedup",
        ),
    )
