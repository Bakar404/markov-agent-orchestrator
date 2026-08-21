"""Simulation run endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_session
from ..schemas import RunCreate, RunResetRequest, RunStatusUpdate, RunStepRequest
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
