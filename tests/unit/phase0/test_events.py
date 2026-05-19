"""Phase 0 — Pydantic event schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from afml.core.events import (
    BarGenerated,
    BetSized,
    Channel,
    ConceptDriftAlert,
    EmergencyFlatten,
    Event,
    MarketRegimeBreak,
    OrderDispatched,
    StrategyValidated,
)


@pytest.mark.phase0
def test_event_envelope_immutable() -> None:
    """Events are frozen so they're safe to share across agent tasks."""
    e = BarGenerated(
        producer="agent_1",
        asset="EURUSD",
        bar_type="tick_imbalance",
        bar_count=100_000,
        jarque_bera_stat=12.3,
    )
    with pytest.raises(ValidationError):
        e.bar_count = 1  # type: ignore[misc]


@pytest.mark.phase0
def test_event_envelope_defaults_populated() -> None:
    e = BarGenerated(
        producer="agent_1",
        asset="EURUSD",
        bar_type="time",
        bar_count=10,
        jarque_bera_stat=1.0,
    )
    assert isinstance(e.event_id, UUID)
    assert isinstance(e.occurred_at, datetime)
    assert e.schema_version == 1
    assert e.channel is Channel.BAR_GENERATED


@pytest.mark.phase0
def test_event_carries_correct_channel() -> None:
    e = EmergencyFlatten(
        producer="control_plane",
        signed_token="abc",
        reason="manual stop",
    )
    assert e.channel is Channel.EMERGENCY_FLATTEN


@pytest.mark.phase0
def test_event_rejects_extra_fields() -> None:
    """Schema is frozen — typos in producers must fail loudly."""
    with pytest.raises(ValidationError):
        StrategyValidated(
            producer="agent_6",
            asset="EURUSD",
            experiment_id=uuid4(),
            pbo=0.03,
            dsr=1.4,
            fwer_penalty=0.2,
            approved=True,
            unknown_field=42,  # type: ignore[call-arg]
        )


@pytest.mark.phase0
def test_discriminated_union_roundtrip() -> None:
    """Serializing then deserializing via the discriminated union recovers the
    original concrete subclass — critical for typed subscribers."""
    adapter: TypeAdapter[Event] = TypeAdapter(Event)
    originals: list[Event] = [
        BarGenerated(
            producer="a1", asset="EURUSD", bar_type="time", bar_count=1, jarque_bera_stat=0.5
        ),
        BetSized(producer="a7", asset="EURUSD", probability=0.62, bet_size=0.5, side="long"),
        OrderDispatched(producer="a7", asset="EURUSD", side="long", size=0.5),
        MarketRegimeBreak(
            producer="a8", asset="EURUSD", test="gsadf", statistic=2.5, critical_value=2.1
        ),
        ConceptDriftAlert(producer="a8", asset="EURUSD", spearman_rank_corr=0.3),
    ]
    for orig in originals:
        wire = orig.model_dump(mode="json")
        roundtripped = adapter.validate_python(wire)
        assert type(roundtripped) is type(orig)
        assert roundtripped == orig


@pytest.mark.phase0
def test_invalid_side_rejected() -> None:
    with pytest.raises(ValidationError):
        BetSized(
            producer="a7",
            asset="EURUSD",
            probability=0.6,
            bet_size=0.5,
            side="diagonal",
        )
