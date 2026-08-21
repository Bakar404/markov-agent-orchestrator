"""Curated offline corpus.

Backs the Research Library when the network is unavailable and provides the citation edges
that make the graph non-empty on a cold start. Every network provider falls back to this
module rather than failing, which is what makes the platform demo-able offline.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache

from ..config import get_settings
from .base import PaperRecord, ProviderResponse, ResearchProvider
from .taxonomy import PROJECT_QUERY, classify, relevance_score, tokenize


@lru_cache(maxsize=1)
def _load() -> dict:
    path = get_settings().seed_corpus_path
    if not path.exists():
        return {"papers": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def corpus_records() -> tuple[PaperRecord, ...]:
    payload = _load()
    records: list[PaperRecord] = []
    for entry in payload.get("papers", []):
        title = entry["title"]
        abstract = entry.get("summary", "")
        tags = classify(title, abstract, entry.get("tags"))
        records.append(
            PaperRecord(
                external_id=entry.get("external_id") or entry["key"],
                source="local_corpus",
                title=title,
                abstract=abstract,
                authors=list(entry.get("authors", [])),
                year=entry.get("year"),
                venue=entry.get("venue", ""),
                url=entry.get("url", ""),
                pdf_url=entry.get("pdf_url", ""),
                citation_count=int(entry.get("citation_count", 0)),
                relevance=relevance_score(
                    title=title,
                    abstract=abstract,
                    query=PROJECT_QUERY,
                    year=entry.get("year"),
                    citation_count=int(entry.get("citation_count", 0)),
                    tags=tags,
                ),
                tags=tags,
                references=list(entry.get("references", [])),
                notes=entry.get("key", ""),
                discovered_via="curated corpus",
            )
        )
    return tuple(records)


@lru_cache(maxsize=1)
def corpus_key_index() -> dict[str, PaperRecord]:
    """``key`` (stored in ``notes``) → record, used to resolve citation edges."""
    return {record.notes: record for record in corpus_records() if record.notes}


def corpus_edges() -> list[tuple[str, str]]:
    """Directed ``(citing_key, cited_key)`` pairs from the curated lineage."""
    index = corpus_key_index()
    edges: list[tuple[str, str]] = []
    for record in corpus_records():
        for reference in record.references:
            if reference in index:
                edges.append((record.notes, reference))
    return edges


def search_corpus(query: str, limit: int = 10) -> list[PaperRecord]:
    """Token-overlap ranking over title, summary, venue, authors and tags."""
    query_tokens = tokenize(query)
    scored: list[tuple[float, PaperRecord]] = []
    for record in corpus_records():
        haystack = " ".join(
            [record.title, record.abstract, record.venue, " ".join(record.authors), " ".join(record.tags)]
        )
        doc_tokens = tokenize(haystack)
        if not query_tokens:
            overlap = 0.5
        else:
            overlap = len(query_tokens & doc_tokens) / len(query_tokens)
        tag_bonus = 0.25 if any(tokenize(tag) & query_tokens for tag in record.tags) else 0.0
        score = overlap + tag_bonus
        if score <= 0.0:
            continue
        clone = PaperRecord(**record.to_dict())
        clone.relevance = relevance_score(
            title=record.title,
            abstract=record.abstract,
            query=query,
            year=record.year,
            citation_count=record.citation_count,
            tags=record.tags,
        )
        scored.append((score, clone))

    scored.sort(key=lambda item: (-item[0], -item[1].relevance))
    return [record for _, record in scored[:limit]]


class LocalCorpusProvider(ResearchProvider):
    id = "local_corpus"
    label = "Curated Corpus"
    kind = "local"
    description = (
        "Offline library of foundational Markov-game, MARL, bandit, planning and agent "
        "orchestration references shipped with the project."
    )

    async def search(self, query: str, limit: int = 10) -> ProviderResponse:
        started = time.perf_counter()
        papers = search_corpus(query, limit)
        for paper in papers:
            paper.discovered_via = f"local_corpus:{query}"
        return ProviderResponse(
            provider=self.id,
            query=query,
            papers=papers,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def health(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "available": True,
            "records": len(corpus_records()),
        }
