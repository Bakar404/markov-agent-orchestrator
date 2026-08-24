"""API routers."""

from __future__ import annotations

from .campaign import router as campaign_router
from .experiments import router as experiments_router
from .meta import router as meta_router
from .profiles import router as profiles_router
from .research import router as research_router
from .runs import router as runs_router
from .ws import router as ws_router

__all__ = [
    "campaign_router",
    "experiments_router",
    "meta_router",
    "profiles_router",
    "research_router",
    "runs_router",
    "ws_router",
]
