"""API routers."""

from __future__ import annotations

from .campaign import router as campaign_router
from .meta import router as meta_router
from .research import router as research_router
from .runs import router as runs_router
from .ws import router as ws_router

__all__ = ["campaign_router", "meta_router", "research_router", "runs_router", "ws_router"]
