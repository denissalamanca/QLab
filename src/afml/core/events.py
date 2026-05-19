"""Pydantic event schemas for inter-agent communication on the message bus.

Every payload published to Redis MUST be one of these models. The discriminated
``Event`` union enables fully typed round-trip deserialization on the subscriber
side. Bumping ``schema_version`` is a breaking change requiring coordinated agent
deploys.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------------
# Channels (Redis pub/sub topics).
# ---------------------------------------------------------------------------------
class Channel(StrEnum):
    BAR_GENERATED = "afml.bar.generated"
    EVENT_TRIGGERED = "afml.event.triggered"
    LABEL_COMPUTED = "afml.label.computed"
    MODEL_TRAINED = "afml.model.trained"
    STRATEGY_VALIDATED = "afml.strategy.validated"
    BET_SIZED = "afml.bet.sized"
    ORDER_DISPATCHED = "afml.order.dispatched"
    MARKET_REGIME_BREAK = "afml.alert.regime_break"
    CONCEPT_DRIFT_ALERT = "afml.alert.concept_drift"
    CEO_APPROVAL = "afml.ceo.approval"
    EMERGENCY_FLATTEN = "afml.ceo.emergency_flatten"
    AGENT_HEARTBEAT = "afml.agent.heartbeat"


# ---------------------------------------------------------------------------------
# Base envelope.
# ---------------------------------------------------------------------------------
def _utcnow() -> datetime:
    return datetime.now(UTC)


class EventEnvelope(BaseModel):
    """Common metadata stamped on every event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(default_factory=uuid4)
    schema_version: Literal[1] = 1
    occurred_at: datetime = Field(default_factory=_utcnow)
    producer: str  # e.g. "agent_1", "control_plane"
    channel: Channel


# ---------------------------------------------------------------------------------
# Concrete events.
# ---------------------------------------------------------------------------------
class BarGenerated(EventEnvelope):
    channel: Literal[Channel.BAR_GENERATED] = Channel.BAR_GENERATED
    asset: str
    bar_type: Literal["time", "tick_imbalance", "tick_run"]
    bar_count: int
    jarque_bera_stat: float


class EventTriggered(EventEnvelope):
    channel: Literal[Channel.EVENT_TRIGGERED] = Channel.EVENT_TRIGGERED
    asset: str
    algorithmic_family: str
    event_count: int
    primary_recall: float | None = None


class LabelComputed(EventEnvelope):
    channel: Literal[Channel.LABEL_COMPUTED] = Channel.LABEL_COMPUTED
    asset: str
    n_events: int
    n_positive: int
    n_negative: int


class ModelTrained(EventEnvelope):
    channel: Literal[Channel.MODEL_TRAINED] = Channel.MODEL_TRAINED
    asset: str
    model_family: str
    brier_score: float
    log_loss: float
    artifact_uri: str


class StrategyValidated(EventEnvelope):
    channel: Literal[Channel.STRATEGY_VALIDATED] = Channel.STRATEGY_VALIDATED
    asset: str
    experiment_id: UUID
    pbo: float
    dsr: float
    fwer_penalty: float
    approved: bool


class BetSized(EventEnvelope):
    channel: Literal[Channel.BET_SIZED] = Channel.BET_SIZED
    asset: str
    probability: float
    bet_size: float
    side: Literal["long", "short", "flat"]


class OrderDispatched(EventEnvelope):
    channel: Literal[Channel.ORDER_DISPATCHED] = Channel.ORDER_DISPATCHED
    asset: str
    side: Literal["long", "short"]
    size: float
    broker_order_id: str | None = None


class MarketRegimeBreak(EventEnvelope):
    channel: Literal[Channel.MARKET_REGIME_BREAK] = Channel.MARKET_REGIME_BREAK
    asset: str
    test: Literal["gsadf", "chow"]
    statistic: float
    critical_value: float


class ConceptDriftAlert(EventEnvelope):
    channel: Literal[Channel.CONCEPT_DRIFT_ALERT] = Channel.CONCEPT_DRIFT_ALERT
    asset: str
    spearman_rank_corr: float


class CEOApproval(EventEnvelope):
    channel: Literal[Channel.CEO_APPROVAL] = Channel.CEO_APPROVAL
    experiment_id: UUID
    signed_token: str
    totp_code: str


class EmergencyFlatten(EventEnvelope):
    channel: Literal[Channel.EMERGENCY_FLATTEN] = Channel.EMERGENCY_FLATTEN
    signed_token: str
    reason: str


class AgentHeartbeat(EventEnvelope):
    channel: Literal[Channel.AGENT_HEARTBEAT] = Channel.AGENT_HEARTBEAT
    agent: str
    status: Literal["healthy", "degraded", "halted"]
    metrics: dict[str, float] = Field(default_factory=dict)


# Discriminated union — enables ``TypeAdapter(Event).validate_python(payload)``
# to deserialize any Event subclass into the correct concrete type.
Event = Annotated[
    BarGenerated
    | EventTriggered
    | LabelComputed
    | ModelTrained
    | StrategyValidated
    | BetSized
    | OrderDispatched
    | MarketRegimeBreak
    | ConceptDriftAlert
    | CEOApproval
    | EmergencyFlatten
    | AgentHeartbeat,
    Field(discriminator="channel"),
]
