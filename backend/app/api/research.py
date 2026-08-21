"""Research Library endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_session
from ..research.service import ResearchService
from ..schemas import ResearchDiscoverRequest, ResearchSearchRequest

router = APIRouter(prefix="/api/research", tags=["research"])


def service(session: Session = Depends(get_session)) -> ResearchService:
    return ResearchService(session)


@router.get("/providers")
async def providers(svc: ResearchService = Depends(service)) -> list[dict]:
    return await svc.provider_health()


@router.post("/search")
async def search(
    payload: ResearchSearchRequest, svc: ResearchService = Depends(service)
) -> dict:
    try:
        return await svc.search(
            payload.query,
            provider_ids=payload.providers,
            limit=payload.limit,
            persist=payload.persist,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/discover")
async def discover(
    payload: ResearchDiscoverRequest | None = None, svc: ResearchService = Depends(service)
) -> dict:
    body = payload or ResearchDiscoverRequest()
    try:
        return await svc.discover(
            categories=body.categories,
            provider_ids=body.providers,
            limit_per_query=body.limit_per_query,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/papers")
def papers(
    tag: str | None = None,
    source: str | None = None,
    search: str | None = None,
    min_relevance: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    svc: ResearchService = Depends(service),
) -> list[dict]:
    return svc.list_papers(
        tag=tag,
        source=source,
        search=search,
        min_relevance=min_relevance,
        limit=limit,
        offset=offset,
    )


@router.get("/papers/{paper_id}")
def paper(paper_id: str, svc: ResearchService = Depends(service)) -> dict:
    try:
        return svc.get_paper(paper_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/papers/{paper_id}/related")
async def related(
    paper_id: str, limit: int = Query(8, ge=1, le=30), svc: ResearchService = Depends(service)
) -> dict:
    try:
        return await svc.related(paper_id, limit=limit)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/graph")
def graph(
    limit: int = Query(220, ge=10, le=1000), svc: ResearchService = Depends(service)
) -> dict:
    return svc.citation_graph(limit=limit)


@router.get("/stats")
def stats(svc: ResearchService = Depends(service)) -> dict:
    return svc.library_stats()


@router.get("/hits")
def hits(limit: int = Query(10, ge=1, le=40), svc: ResearchService = Depends(service)) -> dict:
    return svc.hits_report(limit=limit)


@router.get("/log")
def provider_log(
    limit: int = Query(50, ge=1, le=500), svc: ResearchService = Depends(service)
) -> list[dict]:
    return svc.provider_log(limit=limit)
