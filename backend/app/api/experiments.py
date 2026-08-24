"""Experiment endpoints — compare a control arm against orchestration arms on the same task."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Run, RunVerdict
from ..schemas import VerdictCreate
from ..services.experiment_service import ExperimentService

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def service(session: Session = Depends(get_session)) -> ExperimentService:
    return ExperimentService(session)


@router.get("")
def list_experiments(svc: ExperimentService = Depends(service)) -> list[dict]:
    return svc.list_experiments()


@router.get("/{experiment}")
def compare(experiment: str, svc: ExperimentService = Depends(service)) -> dict:
    try:
        return svc.compare(experiment)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
