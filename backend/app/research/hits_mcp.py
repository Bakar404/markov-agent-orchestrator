"""HITS MCP research provider.

A pluggable external research source treated as an MCP tool server. When
``HITS_MCP_ENDPOINT`` is configured the provider issues a JSON-RPC 2.0 ``tools/call`` against
that endpoint and normalizes whatever the tool returns into :class:`PaperRecord` values, so a
new MCP research backend can be attached without touching the rest of the platform.

When no endpoint is configured the provider stays useful instead of disappearing: it ranks the
curated corpus with Kleinberg's HITS algorithm over the citation graph, surfacing the
*authorities* (heavily-cited foundations) and *hubs* (surveys and syntheses) for a topic. That
is the same hub/authority signal the remote tool is expected to provide, computed locally.
"""

from __future__ import annotations

import json
import time
import uuid

import httpx
import numpy as np

from ..config import get_settings
from .base import PaperRecord, ProviderResponse, ResearchProvider
from .corpus import corpus_edges, corpus_key_index, corpus_records, search_corpus
from .taxonomy import classify, relevance_score, tokenize


def hits_scores(
    node_keys: list[str], edges: list[tuple[str, str]], iterations: int = 40
) -> tuple[dict[str, float], dict[str, float]]:
    """Kleinberg HITS. Returns ``(hubs, authorities)`` normalized to unit max.

    A node is a good *authority* when many hubs cite it, and a good *hub* when it cites many
    authorities. On a citation graph that separates foundational papers from surveys.
    """
    index = {key: i for i, key in enumerate(node_keys)}
    n = len(node_keys)
    if n == 0:
        return {}, {}

    adjacency = np.zeros((n, n), dtype=float)
    for citing, cited in edges:
        if citing in index and cited in index:
            adjacency[index[citing], index[cited]] = 1.0

    hubs = np.ones(n, dtype=float)
    authorities = np.ones(n, dtype=float)
    for _ in range(iterations):
        new_authorities = adjacency.T @ hubs
        new_hubs = adjacency @ new_authorities
        a_norm = np.linalg.norm(new_authorities)
        h_norm = np.linalg.norm(new_hubs)
        authorities = new_authorities / a_norm if a_norm > 0 else new_authorities
        hubs = new_hubs / h_norm if h_norm > 0 else new_hubs

    def scale(vector: np.ndarray) -> dict[str, float]:
        peak = float(vector.max()) if vector.size else 0.0
        if peak <= 0:
            return {key: 0.0 for key in node_keys}
        return {key: float(vector[index[key]] / peak) for key in node_keys}

    return scale(hubs), scale(authorities)


class HITSMCPResearchProvider(ResearchProvider):
    id = "hits_mcp"
    label = "HITS MCP Research"
    kind = "mcp"
    description = (
        "External MCP research source. Calls a JSON-RPC tool server when configured; otherwise "
        "ranks the local corpus with Kleinberg HITS hub/authority scores over the citation graph."
    )
    homepage = ""

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def configured(self) -> bool:
        return bool(self.settings.hits_mcp_endpoint)

    # -------------------------------------------------------------- interface
    async def search(self, query: str, limit: int = 10) -> ProviderResponse:
        started = time.perf_counter()
        if not self.configured or not self.settings.research_allow_network:
            reason = "no MCP endpoint configured" if not self.configured else "network disabled"
            return self._local_hits(query, limit, started, reason)

        try:
            payload = await self._call_tool(
                self.settings.hits_mcp_tool, {"query": query, "limit": limit}
            )
            papers = self._normalize(payload, query, limit)
        except Exception as exc:
            return self._local_hits(query, limit, started, f"{type(exc).__name__}: {exc}")

        if not papers:
            return self._local_hits(query, limit, started, "MCP tool returned no records")

        return ProviderResponse(
            provider=self.id,
            query=query,
            papers=papers,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def related(self, paper: PaperRecord, limit: int = 8) -> ProviderResponse:
        """Discover related work by combining the paper's tags with its title."""
        query = f"{paper.title} {' '.join(paper.tags)}"
        return await self.search(query, limit=limit)

    async def health(self) -> dict:
        base = {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "configured": self.configured,
            "endpoint": self.settings.hits_mcp_endpoint or "(local HITS fallback)",
            "tool": self.settings.hits_mcp_tool,
        }
        if not self.configured or not self.settings.research_allow_network:
            base["available"] = True
            base["mode"] = "local-hits"
            return base
        try:
            payload = await self._rpc("tools/list", {})
            tools = [t.get("name") for t in (payload.get("tools") or [])]
            base.update({"available": True, "mode": "mcp", "tools": tools})
        except Exception as exc:
            base.update({"available": True, "mode": "local-hits", "error": str(exc)})
        return base

    # ----------------------------------------------------------------- MCP
    async def _rpc(self, method: str, params: dict) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.settings.hits_mcp_api_key:
            headers["Authorization"] = f"Bearer {self.settings.hits_mcp_api_key}"
        body = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": method,
            "params": params,
        }
        async with httpx.AsyncClient(timeout=self.settings.research_http_timeout) as client:
            response = await client.post(
                self.settings.hits_mcp_endpoint, json=body, headers=headers
            )
            response.raise_for_status()
            payload = response.json()
        if "error" in payload:
            raise RuntimeError(str(payload["error"]))
        return payload.get("result") or {}

    async def _call_tool(self, name: str, arguments: dict) -> dict:
        return await self._rpc("tools/call", {"name": name, "arguments": arguments})

    def _normalize(self, result: dict, query: str, limit: int) -> list[PaperRecord]:
        """Accept either MCP ``structuredContent`` or JSON embedded in text content blocks."""
        items: list[dict] = []
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            items = structured.get("papers") or structured.get("results") or []
        elif isinstance(structured, list):
            items = structured

        if not items:
            for block in result.get("content") or []:
                if block.get("type") != "text":
                    continue
                try:
                    parsed = json.loads(block.get("text", ""))
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    items = parsed.get("papers") or parsed.get("results") or []
                elif isinstance(parsed, list):
                    items = parsed
                if items:
                    break

        papers: list[PaperRecord] = []
        for item in items[:limit]:
            title = item.get("title") or ""
            if not title:
                continue
            abstract = item.get("abstract") or item.get("summary") or ""
            year = item.get("year")
            citation_count = int(item.get("citation_count") or item.get("citationCount") or 0)
            tags = classify(title, abstract, item.get("tags"))
            papers.append(
                PaperRecord(
                    external_id=str(item.get("id") or item.get("external_id") or title),
                    source=self.id,
                    title=title,
                    abstract=abstract,
                    authors=list(item.get("authors") or []),
                    year=int(year) if isinstance(year, (int, str)) and str(year).isdigit() else None,
                    venue=item.get("venue") or "",
                    url=item.get("url") or "",
                    pdf_url=item.get("pdf_url") or "",
                    citation_count=citation_count,
                    relevance=float(
                        item.get("relevance")
                        or relevance_score(
                            title=title,
                            abstract=abstract,
                            query=query,
                            year=year if isinstance(year, int) else None,
                            citation_count=citation_count,
                            tags=tags,
                        )
                    ),
                    tags=tags,
                    references=list(item.get("references") or []),
                    discovered_via=f"hits_mcp:{query}",
                )
            )
        return papers

    # ---------------------------------------------------------- local HITS
    def _local_hits(
        self, query: str, limit: int, started: float, reason: str
    ) -> ProviderResponse:
        records = corpus_records()
        keys = [r.notes for r in records if r.notes]
        hubs, authorities = hits_scores(keys, corpus_edges())

        query_tokens = tokenize(query)
        scored: list[tuple[float, PaperRecord]] = []
        for record in records:
            key = record.notes
            authority = authorities.get(key, 0.0)
            hub = hubs.get(key, 0.0)
            doc_tokens = tokenize(f"{record.title} {record.abstract} {' '.join(record.tags)}")
            overlap = (
                len(query_tokens & doc_tokens) / len(query_tokens) if query_tokens else 0.4
            )
            score = 0.5 * overlap + 0.35 * authority + 0.15 * hub
            if score <= 0.0:
                continue
            clone = PaperRecord(**record.to_dict())
            clone.source = self.id
            clone.relevance = round(min(score, 1.0), 4)
            clone.discovered_via = f"hits_mcp(local):{query}"
            clone.notes = (
                f"HITS authority={authority:.3f} hub={hub:.3f}"
            )
            scored.append((score, clone))

        scored.sort(key=lambda item: -item[0])
        papers = [record for _, record in scored[:limit]]
        if not papers:
            papers = search_corpus(query, limit)

        return ProviderResponse(
            provider=self.id,
            query=query,
            papers=papers,
            degraded=True,
            error=reason,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def hub_authority_report(self, limit: int = 10) -> dict:
        """Exposed on the API so the UI can show why a paper ranks where it does."""
        records = corpus_records()
        index = corpus_key_index()
        keys = [r.notes for r in records if r.notes]
        hubs, authorities = hits_scores(keys, corpus_edges())
        top_authorities = sorted(authorities.items(), key=lambda kv: -kv[1])[:limit]
        top_hubs = sorted(hubs.items(), key=lambda kv: -kv[1])[:limit]
        return {
            "algorithm": "Kleinberg HITS over the Research Library citation graph",
            "authorities": [
                {"key": k, "title": index[k].title if k in index else k, "score": round(v, 4)}
                for k, v in top_authorities
            ],
            "hubs": [
                {"key": k, "title": index[k].title if k in index else k, "score": round(v, 4)}
                for k, v in top_hubs
            ],
        }
