"""Pydantic request/response schemas for the control-plane API (§11.1).

These are the wire contracts the React frontend talks to. They are
deliberately separate from the internal :class:`afml.core.registry.Experiment`
ORM model (exposed via ``from_attributes``) and from the inter-agent
:mod:`afml.core.events` bus models.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StrategyOut(BaseModel):
    """A validated strategy awaiting CEO sign-off (GET /registry/strategies)."""

    model_config = ConfigDict(from_attributes=True)

    experiment_id: UUID
    asset: str
    algorithmic_family: str
    agent_version: str
    timestamp: datetime
    num_events_triggered: int
    orthogonality_score: float | None
    brain_1_recall: float | None
    brain_2_log_loss: float | None
    pbo: float | None
    dsr: float | None
    is_deployed: bool
    status: str


class ApproveRequest(BaseModel):
    """Body for POST /execution/approve.

    ``signed_token`` is the hex Ed25519 signature over the canonical message
    ``afml:approve:<experiment_id>`` (see :func:`afml.crypto.approval_message`).
    ``totp_code`` is the live 6-digit second factor.
    """

    experiment_id: UUID
    signed_token: str
    totp_code: str


class ApproveResponse(BaseModel):
    experiment_id: UUID
    deployed: bool
    message: str


class FlattenRequest(BaseModel):
    """Body for POST /emergency/flatten.

    ``signed_token`` is the hex Ed25519 signature over ``afml:flatten:<nonce>``
    (see :func:`afml.crypto.flatten_message`). ``reference_prices`` maps each
    open asset to its current price; omitted assets close at entry price.
    """

    nonce: str
    signed_token: str
    reason: str
    reference_prices: dict[str, float] = Field(default_factory=dict)


class FlattenFill(BaseModel):
    """A single closing fill returned by the flatten endpoint."""

    asset: str
    side: str
    size: float
    fill_price: float
    status: str


class FlattenResponse(BaseModel):
    flattened: bool
    n_positions_closed: int
    reason: str
    fills: list[FlattenFill]


class HealthResponse(BaseModel):
    status: str
    service: str
