"""Control-plane HTTP routes (Blueprint §11.1).

Three endpoints under ``/api/v1``:

- ``GET  /registry/strategies`` — CPCV-validated strategies awaiting sign-off.
- ``POST /execution/approve``    — Ed25519 signature **+** TOTP → Paper→Live.
- ``POST /emergency/flatten``    — Ed25519 signature → close all positions.

All capital-allocating actions verify through :class:`CEOAuthenticator`; any
authorisation failure becomes an HTTP 403.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from afml.control_plane.deps import ControlPlaneDeps, get_deps
from afml.control_plane.schemas import (
    ApproveRequest,
    ApproveResponse,
    FlattenFill,
    FlattenRequest,
    FlattenResponse,
    HealthResponse,
    StrategyOut,
)
from afml.control_plane.security import CEOAuthError
from afml.core.events import CEOApproval, EmergencyFlatten

router = APIRouter(prefix="/api/v1", tags=["control-plane"])

DepsDep = Annotated[ControlPlaneDeps, Depends(get_deps)]

_PRODUCER = "control_plane"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe for the frontend / orchestration."""
    return HealthResponse(status="ok", service="afml-control-plane")


@router.get("/registry/strategies", response_model=list[StrategyOut])
def list_strategies(deps: DepsDep) -> list[StrategyOut]:
    """Return all CPCV-validated strategies awaiting CEO sign-off (§11.1).

    A strategy qualifies when it has cleared validation (``pbo`` / ``dsr``
    populated), carries the ``completed`` status, and is not yet deployed.
    """
    experiments = deps.repository.awaiting_signoff()
    return [StrategyOut.model_validate(exp) for exp in experiments]


@router.post("/execution/approve", response_model=ApproveResponse)
def approve(req: ApproveRequest, deps: DepsDep) -> ApproveResponse:
    """Cryptographically authorise a strategy Paper → Live (§11.1).

    Rejects (HTTP 403) any request whose Ed25519 signature does not verify
    against ``afml:approve:<experiment_id>`` **or** whose TOTP code is invalid.
    On success the experiment is marked deployed and a ``CEOApproval`` event is
    published to the agent bus.
    """
    exp = deps.repository.get(req.experiment_id)
    if exp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no experiment with id={req.experiment_id}",
        )

    try:
        deps.authenticator.verify_approval(req.experiment_id, req.signed_token, req.totp_code)
    except CEOAuthError as err:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(err)) from err

    deps.repository.mark_deployed(req.experiment_id, deployed=True)
    deps.event_publisher.publish(
        CEOApproval(
            producer=_PRODUCER,
            experiment_id=req.experiment_id,
            signed_token=req.signed_token,
            totp_code=req.totp_code,
        )
    )
    return ApproveResponse(
        experiment_id=req.experiment_id,
        deployed=True,
        message="strategy promoted Paper → Live",
    )


@router.post("/emergency/flatten", response_model=FlattenResponse)
def emergency_flatten(req: FlattenRequest, deps: DepsDep) -> FlattenResponse:
    """Liquidate every open position (§11.1 kill-switch).

    Rejects (HTTP 403) any request whose Ed25519 signature does not verify
    against ``afml:flatten:<nonce>`` (TOTP is intentionally *not* required so
    the emergency path is never blocked by a code window). On success it
    propagates to Agent 7 (the :class:`ExecutionEngine`), closing all simulated
    positions, resets the risk budget, and publishes an ``EmergencyFlatten``
    event to the agent bus.
    """
    try:
        deps.authenticator.verify_flatten(req.nonce, req.signed_token)
    except CEOAuthError as err:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(err)) from err

    fills = deps.execution_engine.emergency_flatten(req.reference_prices)
    deps.event_publisher.publish(
        EmergencyFlatten(producer=_PRODUCER, signed_token=req.signed_token, reason=req.reason)
    )
    return FlattenResponse(
        flattened=True,
        n_positions_closed=len(fills),
        reason=req.reason,
        fills=[
            FlattenFill(
                asset=fill.asset,
                side=str(fill.side),
                size=fill.size,
                fill_price=fill.fill_price,
                status=str(fill.status),
            )
            for fill in fills
        ],
    )
