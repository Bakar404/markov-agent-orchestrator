"""Static metadata: agents, actions, policies, taxonomy and reward weights."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_session
from ..orchestration.actions import action_catalog
from ..orchestration.agents import agent_catalog
from ..orchestration.policies import policy_catalog
from ..orchestration.state import FEATURE_NAMES
from ..orchestration.strategies import STRATEGY_INDEX, strategy_catalog
from ..research.service import ResearchService
from ..research.taxonomy import taxonomy_catalog

router = APIRouter(prefix="/api/meta", tags=["meta"])


@router.get("")
def meta() -> dict:
    settings = get_settings()
    return {
        "app": settings.app_name,
        "version": settings.version,
        "agents": agent_catalog(),
        "actions": action_catalog(),
        "policies": policy_catalog(),
        "strategies": strategy_catalog(),
        "taxonomy": taxonomy_catalog(),
        "reward_weights": settings.reward_weights.as_dict(),
        "state_features": list(FEATURE_NAMES),
        "defaults": {
            "seed": settings.default_seed,
            "max_steps": settings.max_steps,
            "belief_dim": settings.belief_dim,
        },
        "research": {
            "network_enabled": settings.research_allow_network,
            "mcp_endpoint_configured": bool(settings.hits_mcp_endpoint),
        },
    }


@router.get("/agents")
def agents() -> list[dict]:
    return agent_catalog()


@router.get("/actions")
def actions() -> list[dict]:
    return action_catalog()


@router.get("/policies")
def policies() -> list[dict]:
    return policy_catalog()


@router.get("/strategies/{strategy_id}/papers")
def strategy_papers(
    strategy_id: str,
    limit: int = Query(3, ge=1, le=20),
    session: Session = Depends(get_session),
) -> dict:
    """The published work a strategy implements, drawn from the live library.

    Papers are the ones tagged with the strategy's taxonomy category, reordered so those
    matching its paper query come first. Relevance ordering survives within ties because
    ``sorted`` is stable.
    """
    strategy = STRATEGY_INDEX.get(strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail=f"Unknown strategy '{strategy_id}'")

    if strategy.is_control:
        return {
            "strategy": strategy.to_dict(),
            "category": strategy.category,
            "paper_query": strategy.paper_query,
            "papers": [],
            "note": (
                "The control implements nothing. It exists so the other arms have something "
                "to beat."
            ),
        }

    candidates = ResearchService(session).list_papers(tag=strategy.category, limit=60)
    terms = [t for t in strategy.paper_query.lower().split() if len(t) > 3]

    def affinity(paper: dict) -> int:
        haystack = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        return sum(term in haystack for term in terms)

    ranked = sorted(candidates, key=affinity, reverse=True)
    return {
        "strategy": strategy.to_dict(),
        "category": strategy.category,
        "paper_query": strategy.paper_query,
        "papers": ranked[:limit],
        "note": None,
    }
