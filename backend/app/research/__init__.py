"""Research Intelligence Layer."""

from __future__ import annotations

from .base import PaperRecord, ProviderResponse, ResearchProvider
from .hits_mcp import HITSMCPResearchProvider
from .registry import DEFAULT_PROVIDERS, get_provider, provider_registry, resolve_providers
from .service import ResearchService
from .taxonomy import CATEGORY_NAMES, taxonomy_catalog

__all__ = [
    "PaperRecord",
    "ProviderResponse",
    "ResearchProvider",
    "HITSMCPResearchProvider",
    "ResearchService",
    "DEFAULT_PROVIDERS",
    "CATEGORY_NAMES",
    "get_provider",
    "provider_registry",
    "resolve_providers",
    "taxonomy_catalog",
]
