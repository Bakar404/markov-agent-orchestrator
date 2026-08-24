"""Experiment endpoints — compare a control arm against orchestration arms on the same task."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import PairwiseVerdict, Run
from ..schemas import PairwiseCreate
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


@router.post("/{experiment}/pairwise", status_code=201)
def record_pairwise(
    experiment: str, payload: PairwiseCreate, session: Session = Depends(get_session)
) -> dict:
    """Record a blind head-to-head preference between two runs.

    Judge the answers without knowing which arm produced which. Absolute scoring compresses
    toward the top of the range; a forced choice does not.
    """
    if payload.run_a == payload.run_b:
        raise HTTPException(status_code=422, detail="run_a and run_b must differ")

    for run_id in (payload.run_a, payload.run_b):
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        if (run.config or {}).get("experiment") != experiment:
            raise HTTPException(
                status_code=422,
                detail=f"Run '{run_id}' is not part of experiment '{experiment}'",
            )

    verdict = PairwiseVerdict(
        experiment=experiment,
        run_a_id=payload.run_a,
        run_b_id=payload.run_b,
        winner=payload.winner,
        judge=payload.judge,
        rubric=payload.rubric,
        notes=payload.notes,
    )
    session.add(verdict)
    session.commit()

    return {
        "experiment": experiment,
        "run_a": payload.run_a,
        "run_b": payload.run_b,
        "winner": payload.winner,
        "judge": payload.judge,
    }
