"""Simulation run endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Run, RunVerdict
from ..schemas import (
    LiveOpenRequest,
    LiveReportRequest,
    RunCreate,
    RunResetRequest,
    RunStatusUpdate,
    RunStepRequest,
    VerdictCreate,
)
from ..services import hub
from ..services.run_service import RunNotFound, RunService

router = APIRouter(prefix="/api/runs", tags=["runs"])


def service(session: Session = Depends(get_session)) -> RunService:
    return RunService(session)


@router.post("", status_code=201)
def create_run(payload: RunCreate, svc: RunService = Depends(service)) -> dict:
    try:
        return svc.create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("")
def list_runs(
    limit: int = Query(50, ge=1, le=200), svc: RunService = Depends(service)
) -> list[dict]:
    return svc.list_runs(limit=limit)


@router.get("/{run_id}")
def get_run(run_id: str, svc: RunService = Depends(service)) -> dict:
    try:
        return svc.detail(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{run_id}/step")
def step_run(
    run_id: str, payload: RunStepRequest | None = None, svc: RunService = Depends(service)
) -> dict:
    steps = payload.steps if payload else 1
    try:
        results = svc.step_many(run_id, steps)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run": svc.detail(run_id), "steps": results}


@router.post("/{run_id}/live/open")
def live_open(
    run_id: str, payload: LiveOpenRequest | None = None, svc: RunService = Depends(service)
) -> dict:
    """Ask the policy who acts next. Returns briefs; does not advance the episode.

    Runs on the ``external`` policy have no choice to ask for — an outside orchestrator made it
    — so those must declare ``action`` and ``agents`` here instead. Every other policy rejects
    a declaration, or the caller could pick the result and call it a policy decision.
    """
    declared = (payload.action, payload.agents) if payload and payload.action else None
    try:
        return svc.live_open(run_id, declared)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{run_id}/live/report")
def live_report(
    run_id: str, payload: LiveReportRequest, svc: RunService = Depends(service)
) -> dict:
    """Fold what the agents actually produced into the episode and advance one step."""
    try:
        result = svc.live_report(
            run_id, payload.token, payload.reports, payload.hypotheses or None
        )
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    detail = svc.detail(run_id)
    # Spectators watch over the WebSocket, which never saw this step because it arrived by REST.
    hub.publish(run_id, {"type": "step", "step": result, "run": detail})
    if result["done"]:
        hub.publish(
            run_id,
            {
                "type": "terminated",
                "run": detail,
                "reason": result["termination_reason"],
            },
        )
    return {"run": detail, "step": result}


@router.post("/{run_id}/live/abandon", status_code=204)
def live_abandon(run_id: str, svc: RunService = Depends(service)) -> None:
    try:
        svc.live_abandon(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{run_id}/verdict", status_code=201)
def score_run(
    run_id: str, payload: VerdictCreate, session: Session = Depends(get_session)
) -> dict:
    """Record how good this run's answer was.

    Cost and tokens are measured automatically; quality cannot be, because the reward function
    pays for belief collapse and would let orchestration win by construction.
    """
    if session.get(Run, run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    existing = session.scalar(
        select(RunVerdict).where(
            RunVerdict.run_id == run_id, RunVerdict.judge == payload.judge
        )
    )
    verdict = existing or RunVerdict(run_id=run_id, judge=payload.judge)
    verdict.score = payload.score
    verdict.rubric = payload.rubric
    verdict.notes = payload.notes
    session.add(verdict)
    session.commit()

    return {
        "run_id": run_id,
        "judge": verdict.judge,
        "score": verdict.score,
        "rubric": verdict.rubric,
    }


@router.post("/{run_id}/reset")
def reset_run(
    run_id: str, payload: RunResetRequest | None = None, svc: RunService = Depends(service)
) -> dict:
    body = payload or RunResetRequest()
    try:
        return svc.reset(
            run_id, seed=body.seed, keep_policy_learning=body.keep_policy_learning
        )
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{run_id}/status")
def set_status(
    run_id: str, payload: RunStatusUpdate, svc: RunService = Depends(service)
) -> dict:
    try:
        return svc.set_status(run_id, payload.status)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{run_id}", status_code=204)
def delete_run(run_id: str, svc: RunService = Depends(service)) -> None:
    try:
        svc.delete(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{run_id}/traces")
def traces(
    run_id: str, limit: int = Query(500, ge=1, le=5000), svc: RunService = Depends(service)
) -> list[dict]:
    try:
        return svc.traces(run_id, limit=limit)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{run_id}/states")
def states(
    run_id: str, limit: int = Query(500, ge=1, le=5000), svc: RunService = Depends(service)
) -> list[dict]:
    try:
        return svc.states(run_id, limit=limit)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{run_id}/messages")
def messages(
    run_id: str, limit: int = Query(1000, ge=1, le=10000), svc: RunService = Depends(service)
) -> list[dict]:
    try:
        return svc.messages(run_id, limit=limit)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{run_id}/metrics")
def metrics(run_id: str, svc: RunService = Depends(service)) -> dict:
    try:
        return svc.metrics(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{run_id}/graph")
def interaction_graph(run_id: str, svc: RunService = Depends(service)) -> dict:
    try:
        return svc.interaction_graph(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
