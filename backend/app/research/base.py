"""Research provider abstraction.

Every source — arXiv, Semantic Scholar, Papers With Code, or an external MCP tool server —
implements the same contract, so the Research Library never knows where a record came from.
Providers must degrade rather than fail: if the network is unavailable the provider answers
from the curated local corpus and flags the response as ``degraded``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


def normalize_title(title: str) -> str:
    """Canonical form used for cross-provider deduplication."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


@dataclass
class PaperRecord:
    """Provider-agnostic paper representation."""

    external_id: str
    source: str
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    url: str = ""
    pdf_url: str = ""
    citation_count: int = 0
    relevance: float = 0.0
    tags: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    notes: str = ""
    discovered_via: str = ""

    @property
    def dedupe_key(self) -> str:
        return normalize_title(self.title)

    def to_dict(self) -> dict:
        return {
            "external_id": self.external_id,
            "source": self.source,
            "title": self.title,
            "abstract": self.abstract,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "citation_count": self.citation_count,
            "relevance": self.relevance,
            "tags": list(self.tags),
            "references": list(self.references),
            "notes": self.notes,
            "discovered_via": self.discovered_via,
        }


@dataclass
class ProviderResponse:
    provider: str
    query: str
    papers: list[PaperRecord]
    degraded: bool = False
    error: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "query": self.query,
            "count": len(self.papers),
            "degraded": self.degraded,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


class ResearchProvider(ABC):
    """Contract implemented by every research source."""

    id: str = "provider"
    label: str = "Provider"
    kind: str = "api"
    description: str = ""
    homepage: str = ""

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> ProviderResponse:
        """Return papers matching ``query``. Must never raise for network reasons."""

    async def related(self, paper: PaperRecord, limit: int = 8) -> ProviderResponse:
        """Discover related work for a paper. Defaults to a title-based search."""
        return await self.search(paper.title, limit=limit)

    async def health(self) -> dict:
        return {"id": self.id, "label": self.label, "kind": self.kind, "available": True}

    def metadata(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "description": self.description,
            "homepage": self.homepage,
        }
