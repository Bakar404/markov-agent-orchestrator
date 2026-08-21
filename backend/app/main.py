"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import campaign_router, meta_router, research_router, runs_router, ws_router
from .config import get_settings
from .db import init_db, session_scope
from .research.service import ResearchService

logger = logging.getLogger("markov_orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with session_scope() as session:
        service = ResearchService(session)
        seeded = service.seed_if_empty()
        rescored = service.backfill_relevance()
    if seeded:
        logger.info("Seeded Research Library with %s curated papers", seeded)
    if rescored:
        logger.info("Backfilled relevance for %s papers", rescored)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Stochastic multi-agent orchestration engine: contextual bandit → MDP → cooperative "
            "Markov game → multi-agent RL, with an attached Research Intelligence Layer."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins) or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(meta_router)
    app.include_router(runs_router)
    app.include_router(research_router)
    app.include_router(campaign_router)
    app.include_router(ws_router)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "app": settings.app_name, "version": settings.version}

    @app.get("/", tags=["meta"])
    def root() -> dict:
        return {
            "name": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "endpoints": ["/api/meta", "/api/runs", "/api/research", "/api/campaign", "/ws/runs/{run_id}"],
        }

    return app


app = create_app()
