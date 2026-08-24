"""Policy profile endpoints — inspect and reset the router's learned parameters."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_session
from ..schemas import PolicyProfileReset
from ..services.policy_profile_service import PolicyProfileService

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


def service(session: Session = Depends(get_session)) -> PolicyProfileService:
    return PolicyProfileService(session)


@router.get("")
def list_profiles(svc: PolicyProfileService = Depends(service)) -> list[dict]:
    return svc.list_profiles()


@router.delete("", status_code=204)
def reset_profile(
    payload: PolicyProfileReset, svc: PolicyProfileService = Depends(service)
) -> None:
    """Discard learned parameters. The bandit measurably degrades when it carries stale ones."""
    if not svc.reset(payload.name, payload.policy):
        raise HTTPException(
            status_code=404, detail=f"No profile '{payload.name}' for policy '{payload.policy}'"
        )
