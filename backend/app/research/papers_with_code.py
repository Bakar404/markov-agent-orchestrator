"""Papers With Code provider.

Adds the implementation dimension: which papers have reproducible code, which is the signal
this project uses to prioritize algorithms that are actually portable into the engine.
"""

from __future__ import annotations

import time

import httpx

from ..config import get_settings
from .base import PaperRecord, ProviderResponse, ResearchProvider
from .corpus import search_corpus
from .taxonomy import classify, relevance_score


class PapersWithCodeProvider(ResearchProvider):
    id = "papers_with_code"
    label = "Papers With Code"
    kind = "api"
    description = "Links papers to open-source implementations, benchmarks and leaderboards."
    homepage = "https://paperswithcode.com"

    def __init__(self) -> None:
        self.settings = get_settings()

    async def search(self, query: str, limit: int = 10) -> ProviderResponse:
        started = time.perf_counter()
        if not self.settings.research_allow_network:
            return self._fallback(query, limit, started, "network disabled")

        url = f"{self.settings.papers_with_code_base_url}/papers/"
        params = {"q": query, "items_per_page": max(1, min(limit, 50))}
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.research_http_timeout,
                headers={"Accept": "application/json"},
                follow_redirects=True,
            ) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return self._fallback(query, limit, started, f"{type(exc).__name__}: {exc}")

        papers = [self._to_record(item, query) for item in payload.get("results", [])]
        return ProviderResponse(
            provider=self.id,
            query=query,
            papers=[p for p in papers if p.title],
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _to_record(self, item: dict, query: str) -> PaperRecord:
        title = item.get("title") or ""
        abstract = item.get("abstract") or ""
        published = item.get("published") or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        tags = classify(title, abstract)
        arxiv_id = item.get("arxiv_id") or ""
        return PaperRecord(
            external_id=arxiv_id or item.get("id") or title,
            source=self.id,
            title=title,
            abstract=abstract,
            authors=list(item.get("authors") or []),
            year=year,
            venue=item.get("proceeding") or "Papers With Code",
            url=item.get("url_abs") or "",
            pdf_url=item.get("url_pdf") or "",
            citation_count=0,
            relevance=relevance_score(
                title=title,
                abstract=abstract,
                query=query,
                year=year,
                citation_count=0,
                tags=tags,
            ),
            tags=tags,
            notes="has_implementation",
            discovered_via=f"papers_with_code:{query}",
        )

    def _fallback(self, query: str, limit: int, started: float, error: str) -> ProviderResponse:
        papers = search_corpus(query, limit)
        for paper in papers:
            paper.discovered_via = f"papers_with_code(degraded):{query}"
        return ProviderResponse(
            provider=self.id,
            query=query,
            papers=papers,
            degraded=True,
            error=error,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def health(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "available": self.settings.research_allow_network,
            "endpoint": self.settings.papers_with_code_base_url,
        }
