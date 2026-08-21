"""Provider registry.

Providers are constructed once and shared. Adding a source means implementing
:class:`~app.research.base.ResearchProvider` and registering it here.
"""

from __future__ import annotations

from functools import lru_cache

from .arxiv import ArxivProvider
from .base import ResearchProvider
from .corpus import LocalCorpusProvider
from .hits_mcp import HITSMCPResearchProvider
from .papers_with_code import PapersWithCodeProvider
from .semantic_scholar import SemanticScholarProvider

DEFAULT_PROVIDERS: tuple[str, ...] = ("arxiv", "semantic_scholar", "hits_mcp")


@lru_cache(maxsize=1)
def provider_registry() -> dict[str, ResearchProvider]:
    providers: list[ResearchProvider] = [
        ArxivProvider(),
        SemanticScholarProvider(),
        PapersWithCodeProvider(),
        HITSMCPResearchProvider(),
        LocalCorpusProvider(),
    ]
    return {provider.id: provider for provider in providers}


def get_provider(provider_id: str) -> ResearchProvider:
    try:
        return provider_registry()[provider_id]
    except KeyError as exc:
        known = ", ".join(sorted(provider_registry()))
        raise ValueError(f"Unknown provider '{provider_id}'. Available: {known}") from exc


def resolve_providers(provider_ids: list[str] | None) -> list[ResearchProvider]:
    ids = provider_ids or list(DEFAULT_PROVIDERS)
    return [get_provider(pid) for pid in ids]
