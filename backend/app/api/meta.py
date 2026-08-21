"""Static metadata: agents, actions, policies, taxonomy and reward weights."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import get_settings
from ..orchestration.actions import action_catalog
from ..orchestration.agents import agent_catalog
from ..orchestration.policies import policy_catalog
from ..orchestration.state import FEATURE_NAMES
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
