"""Pydantic request/response schemas for the control-plane API (§11.1).

These are the wire contracts the React frontend talks to. They are
deliberately separate from the internal :class:`afml.core.registry.Experiment`
ORM model (exposed via ``from_attributes``) and from the inter-agent
:mod:`afml.core.events` bus models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrategyOut(BaseModel):
    """A validated strategy awaiting CEO sign-off (GET /registry/strategies).

    AFML 0-9 final audit V3: the Alpha Registry / ML pipeline produce
    ``numpy.float64`` / ``numpy.int64`` / ``pandas.Timestamp`` values. Standard
    JSON encoders raise ``Object of type int64 is not JSON serializable``. The
    ``mode="before"`` validators below scrub every field to a native Python
    type (and pandas/numpy datetimes to ``datetime`` → ISO-8601 on dump) at the
    API boundary, so the response can never crash with a serialization error.
    """

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

    @field_validator(
        "orthogonality_score",
        "brain_1_recall",
        "brain_2_log_loss",
        "pbo",
        "dsr",
        mode="before",
    )
    @classmethod
    def _coerce_optional_float(cls, v: Any) -> float | None:
        """numpy.float64 → native float (None passes through)."""
        return None if v is None else float(v)

    @field_validator("num_events_triggered", mode="before")
    @classmethod
    def _coerce_int(cls, v: Any) -> int:
        """numpy.int64 → native int."""
        return int(v)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _coerce_datetime(cls, v: Any) -> Any:
        """pandas.Timestamp / numpy.datetime64 → python datetime (ISO-8601 on dump)."""
        to_pydatetime = getattr(v, "to_pydatetime", None)
        return to_pydatetime() if callable(to_pydatetime) else v


class ApproveRequest(BaseModel):
    """Body for POST /execution/approve.

    ``signed_token`` is the hex Ed25519 signature over the canonical message
    ``afml:approve:<experiment_id>:<timestamp_ms>`` (see
    :func:`afml.crypto.approval_message`). ``timestamp_ms`` is the millisecond
    UTC time the CEO signed at — the backend rejects it outside a ±60 s window
    (audit V1 anti-replay). ``totp_code`` is the live 6-digit second factor.
    """

    experiment_id: UUID
    timestamp_ms: int
    signed_token: str
    totp_code: str


class ApproveResponse(BaseModel):
    experiment_id: UUID
    deployed: bool
    message: str


class FlattenRequest(BaseModel):
    """Body for POST /emergency/flatten.

    ``signed_token`` is the hex Ed25519 signature over
    ``afml:flatten:<nonce>:<timestamp_ms>`` (see
    :func:`afml.crypto.flatten_message`). ``timestamp_ms`` (±60 s window) plus
    the single-use ``nonce`` give layered anti-replay (audit V1).
    ``reference_prices`` maps each open asset to its current price; omitted
    assets close at entry price.
    """

    nonce: str
    timestamp_ms: int
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
