"""arXiv provider (Atom API).

Uses the public export API with no key. Degrades to the curated corpus when the network is
disabled or the request fails.
"""

from __future__ import annotations

import time
from xml.etree import ElementTree

import httpx

from ..config import get_settings
from .base import PaperRecord, ProviderResponse, ResearchProvider
from .corpus import search_corpus
from .taxonomy import classify, relevance_score

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivProvider(ResearchProvider):
    id = "arxiv"
    label = "arXiv"
    kind = "api"
    description = "Open-access preprint server queried through the public Atom export API."
    homepage = "https://arxiv.org"

    def __init__(self) -> None:
        self.settings = get_settings()

    async def search(self, query: str, limit: int = 10) -> ProviderResponse:
        started = time.perf_counter()
        if not self.settings.research_allow_network:
            return self._fallback(query, limit, started, "network disabled")

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max(1, min(limit, 50)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.research_http_timeout) as client:
                response = await client.get(self.settings.arxiv_base_url, params=params)
                response.raise_for_status()
                papers = self._parse(response.text, query)
        except Exception as exc:  # network, parse, or HTTP error - degrade, never fail
            return self._fallback(query, limit, started, f"{type(exc).__name__}: {exc}")

        return ProviderResponse(
            provider=self.id,
            query=query,
            papers=papers,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _parse(self, xml: str, query: str) -> list[PaperRecord]:
        root = ElementTree.fromstring(xml)
        papers: list[PaperRecord] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            raw_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
            external_id = raw_id.rsplit("/", 1)[-1]
            title = " ".join((entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").split())
            summary = " ".join(
                (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").split()
            )
            published = entry.findtext("atom:published", default="", namespaces=ATOM_NS) or ""
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = [
                (node.findtext("atom:name", default="", namespaces=ATOM_NS) or "").strip()
                for node in entry.findall("atom:author", ATOM_NS)
            ]
            pdf_url = ""
            for link in entry.findall("atom:link", ATOM_NS):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", "")
            venue = entry.findtext("arxiv:journal_ref", default="", namespaces=ATOM_NS) or "arXiv"
            tags = classify(title, summary)

            papers.append(
                PaperRecord(
                    external_id=external_id,
                    source=self.id,
                    title=title,
                    abstract=summary,
                    authors=[a for a in authors if a],
                    year=year,
                    venue=venue,
                    url=raw_id,
                    pdf_url=pdf_url,
                    citation_count=0,
                    relevance=relevance_score(
                        title=title,
                        abstract=summary,
                        query=query,
                        year=year,
                        citation_count=0,
                        tags=tags,
                    ),
                    tags=tags,
                    discovered_via=f"arxiv:{query}",
                )
            )
        return papers

    def _fallback(self, query: str, limit: int, started: float, error: str) -> ProviderResponse:
        papers = search_corpus(query, limit)
        for paper in papers:
            paper.discovered_via = f"arxiv(degraded):{query}"
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
            "endpoint": self.settings.arxiv_base_url,
        }
