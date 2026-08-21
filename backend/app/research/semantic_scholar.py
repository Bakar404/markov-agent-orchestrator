"""Semantic Scholar Graph API provider.

Supplies the two things arXiv cannot: real citation counts and real reference lists, which is
what turns the Research Library's citation graph from curated to observed.
"""

from __future__ import annotations

import time

import httpx

from ..config import get_settings
from .base import PaperRecord, ProviderResponse, ResearchProvider
from .corpus import search_corpus
from .taxonomy import classify, relevance_score

SEARCH_FIELDS = (
    "paperId,externalIds,title,abstract,year,venue,citationCount,authors,openAccessPdf,url"
)
REFERENCE_FIELDS = "title,externalIds,year,venue,citationCount,authors,abstract,url"


class SemanticScholarProvider(ResearchProvider):
    id = "semantic_scholar"
    label = "Semantic Scholar"
    kind = "api"
    description = (
        "Academic graph with citation counts and reference edges. An API key raises the rate "
        "limit but is not required."
    )
    homepage = "https://www.semanticscholar.org"

    def __init__(self) -> None:
        self.settings = get_settings()

    def _headers(self) -> dict[str, str]:
        if self.settings.semantic_scholar_api_key:
            return {"x-api-key": self.settings.semantic_scholar_api_key}
        return {}

    async def search(self, query: str, limit: int = 10) -> ProviderResponse:
        started = time.perf_counter()
        if not self.settings.research_allow_network:
            return self._fallback(query, limit, started, "network disabled")

        url = f"{self.settings.semantic_scholar_base_url}/paper/search"
        params = {"query": query, "limit": max(1, min(limit, 100)), "fields": SEARCH_FIELDS}
        try:
            async with httpx.AsyncClient(timeout=self.settings.research_http_timeout) as client:
                response = await client.get(url, params=params, headers=self._headers())
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return self._fallback(query, limit, started, f"{type(exc).__name__}: {exc}")

        papers = [self._to_record(item, query) for item in payload.get("data", [])]
        return ProviderResponse(
            provider=self.id,
            query=query,
            papers=[p for p in papers if p.title],
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def references(self, paper_id: str, limit: int = 15) -> ProviderResponse:
        """Fetch outgoing citation edges for a Semantic Scholar paper id or DOI/arXiv id."""
        started = time.perf_counter()
        if not self.settings.research_allow_network:
            return ProviderResponse(
                provider=self.id, query=paper_id, papers=[], degraded=True, error="network disabled"
            )

        url = f"{self.settings.semantic_scholar_base_url}/paper/{paper_id}/references"
        params = {"limit": max(1, min(limit, 100)), "fields": REFERENCE_FIELDS}
        try:
            async with httpx.AsyncClient(timeout=self.settings.research_http_timeout) as client:
                response = await client.get(url, params=params, headers=self._headers())
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return ProviderResponse(
                provider=self.id,
                query=paper_id,
                papers=[],
                degraded=True,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        papers = []
        for item in payload.get("data", []):
            cited = item.get("citedPaper") or {}
            record = self._to_record(cited, paper_id)
            if record.title:
                papers.append(record)
        return ProviderResponse(
            provider=self.id,
            query=paper_id,
            papers=papers,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _to_record(self, item: dict, query: str) -> PaperRecord:
        external_ids = item.get("externalIds") or {}
        external_id = (
            external_ids.get("ArXiv")
            or external_ids.get("DOI")
            or item.get("paperId")
            or item.get("title", "")
        )
        title = item.get("title") or ""
        abstract = item.get("abstract") or ""
        year = item.get("year")
        citation_count = int(item.get("citationCount") or 0)
        tags = classify(title, abstract)
        authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
        pdf = (item.get("openAccessPdf") or {}).get("url", "") or ""
        return PaperRecord(
            external_id=str(external_id),
            source=self.id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=item.get("venue") or "",
            url=item.get("url") or "",
            pdf_url=pdf,
            citation_count=citation_count,
            relevance=relevance_score(
                title=title,
                abstract=abstract,
                query=query,
                year=year,
                citation_count=citation_count,
                tags=tags,
            ),
            tags=tags,
            discovered_via=f"semantic_scholar:{query}",
        )

    def _fallback(self, query: str, limit: int, started: float, error: str) -> ProviderResponse:
        papers = search_corpus(query, limit)
        for paper in papers:
            paper.discovered_via = f"semantic_scholar(degraded):{query}"
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
            "authenticated": bool(self.settings.semantic_scholar_api_key),
            "endpoint": self.settings.semantic_scholar_base_url,
        }
