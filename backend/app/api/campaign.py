"""Cross-episode learning experiment endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import CampaignRequest
from ..services.campaign_service import run_campaign

router = APIRouter(prefix="/api/campaign", tags=["campaign"])


# Defined sync on purpose: the experiment is CPU-bound numpy, so FastAPI runs it in a
# threadpool rather than blocking the event loop.
@router.post("")
def campaign(payload: CampaignRequest) -> dict:
    try:
        return run_campaign(
            policies=payload.policies,
            episodes=payload.episodes,
            seed_base=payload.seed_base,
            max_steps=payload.max_steps,
            budget_usd=payload.budget_usd,
            task_complexity=payload.task_complexity,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
