"""Research Library service.

Owns everything above the provider layer: fan-out search, cross-provider deduplication,
persistence, tagging, relevance scoring and the citation graph.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import Citation, Paper, PaperTag, ProviderQuery
from .base import PaperRecord, ProviderResponse, normalize_title
from .corpus import corpus_edges, corpus_key_index, corpus_records
from .hits_mcp import HITSMCPResearchProvider, hits_scores
from .registry import get_provider, provider_registry, resolve_providers
from .taxonomy import CATEGORIES, PROJECT_QUERY, relevance_score, taxonomy_catalog


class ResearchService:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------- seeding
    def seed_if_empty(self) -> int:
        """Populate the library from the curated corpus on a cold start."""
        existing = self.session.scalar(select(func.count()).select_from(Paper)) or 0
        if existing:
            return 0

        records = list(corpus_records())
        key_to_id: dict[str, str] = {}
        for record in records:
            paper = self._upsert(record)
            if record.notes:
                key_to_id[record.notes] = paper.id
        self.session.flush()

        for citing_key, cited_key in corpus_edges():
            source_id = key_to_id.get(citing_key)
            target_id = key_to_id.get(cited_key)
            if source_id and target_id:
                self._link(source_id, target_id, relation="cites")

        self.session.flush()
        self._recompute_offline_citation_counts()
        self.session.commit()
        return len(records)

    def backfill_relevance(self) -> int:
        """Score any paper still sitting at zero relevance against the project topic.

        Seeded records predate relevance scoring, so this keeps existing databases ranked
        without forcing a wipe.
        """
        updated = 0
        for paper in self.session.scalars(select(Paper).where(Paper.relevance <= 0.0)).unique():
            paper.relevance = relevance_score(
                title=paper.title,
                abstract=paper.abstract,
                query=PROJECT_QUERY,
                year=paper.year,
                citation_count=paper.citation_count,
                tags=[t.tag for t in paper.tags],
            )
            updated += 1
        if updated:
            self.session.commit()
        return updated

    # --------------------------------------------------------------- search
    async def search(
        self,
        query: str,
        *,
        provider_ids: list[str] | None = None,
        limit: int = 10,
        persist: bool = True,
    ) -> dict:
        providers = resolve_providers(provider_ids)
        responses = await asyncio.gather(
            *(provider.search(query, limit=limit) for provider in providers)
        )

        merged = self._merge(responses)
        stored: list[dict] = []
        if persist:
            papers = [self._upsert(record) for record in merged]
            self.session.flush()
            self._link_reference_titles(merged, papers)
            self._recompute_offline_citation_counts()
            self.session.commit()
            stored = [self._serialize(paper) for paper in papers]
        else:
            stored = [record.to_dict() for record in merged]

        for response in responses:
            self._audit(response)
        self.session.commit()

        return {
            "query": query,
            "providers": [response.to_dict() for response in responses],
            "degraded": any(r.degraded for r in responses),
            "count": len(stored),
            "papers": sorted(stored, key=lambda p: -p.get("relevance", 0.0)),
        }

    async def discover(
        self,
        *,
        categories: list[str] | None = None,
        provider_ids: list[str] | None = None,
        limit_per_query: int = 6,
    ) -> dict:
        """Run the standing discovery queries for each taxonomy category."""
        selected = [c for c in CATEGORIES if not categories or c.name in categories]
        results = []
        total_new = 0
        before = self.session.scalar(select(func.count()).select_from(Paper)) or 0

        for category in selected:
            for query in category.discovery_queries:
                outcome = await self.search(
                    query, provider_ids=provider_ids, limit=limit_per_query
                )
                results.append(
                    {
                        "category": category.name,
                        "query": query,
                        "count": outcome["count"],
                        "degraded": outcome["degraded"],
                    }
                )

        after = self.session.scalar(select(func.count()).select_from(Paper)) or 0
        total_new = after - before
        return {
            "categories": [c.name for c in selected],
            "queries_executed": len(results),
            "new_papers": total_new,
            "library_size": after,
            "results": results,
        }

    async def related(self, paper_id: str, *, limit: int = 8) -> dict:
        paper = self.session.get(Paper, paper_id)
        if paper is None:
            raise LookupError(f"Paper '{paper_id}' not found")
        record = PaperRecord(
            external_id=paper.external_id,
            source=paper.source,
            title=paper.title,
            abstract=paper.abstract,
            tags=[t.tag for t in paper.tags],
        )
        provider = get_provider("hits_mcp")
        response = await provider.related(record, limit=limit)
        discovered = [self._upsert(r) for r in response.papers if r.dedupe_key != record.dedupe_key]
        self.session.flush()
        for target in discovered:
            self._link(paper.id, target.id, relation="related", weight=0.5)
        self._recompute_offline_citation_counts()
        self._audit(response)
        self.session.commit()
        return {
            "paper": self._serialize(paper),
            "provider": response.to_dict(),
            "related": [self._serialize(p) for p in discovered],
        }

    # ------------------------------------------------------------- querying
    def list_papers(
        self,
        *,
        tag: str | None = None,
        source: str | None = None,
        search: str | None = None,
        min_relevance: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        stmt = select(Paper)
        if tag:
            stmt = stmt.join(PaperTag).where(PaperTag.tag == tag)
        if source:
            stmt = stmt.where(Paper.source == source)
        if search:
            pattern = f"%{search.lower()}%"
            stmt = stmt.where(
                func.lower(Paper.title).like(pattern) | func.lower(Paper.abstract).like(pattern)
            )
        if min_relevance > 0:
            stmt = stmt.where(Paper.relevance >= min_relevance)
        stmt = stmt.order_by(Paper.relevance.desc(), Paper.citation_count.desc()).limit(limit).offset(offset)
        return [self._serialize(paper) for paper in self.session.scalars(stmt).unique()]

    def get_paper(self, paper_id: str) -> dict:
        paper = self.session.get(Paper, paper_id)
        if paper is None:
            raise LookupError(f"Paper '{paper_id}' not found")
        payload = self._serialize(paper)
        payload["references"] = [
            self._serialize(p)
            for p in self._neighbors(paper_id, outgoing=True)
        ]
        payload["cited_by"] = [
            self._serialize(p)
            for p in self._neighbors(paper_id, outgoing=False)
        ]
        return payload

    def citation_graph(self, *, limit: int = 220) -> dict:
        papers = list(
            self.session.scalars(
                select(Paper).order_by(Paper.relevance.desc(), Paper.citation_count.desc()).limit(limit)
            ).unique()
        )
        ids = {p.id for p in papers}
        edges = [
            edge
            for edge in self.session.scalars(select(Citation)).unique()
            if edge.source_paper_id in ids and edge.target_paper_id in ids
        ]

        degree_in: dict[str, int] = defaultdict(int)
        degree_out: dict[str, int] = defaultdict(int)
        for edge in edges:
            degree_out[edge.source_paper_id] += 1
            degree_in[edge.target_paper_id] += 1

        hubs, authorities = hits_scores(
            [p.id for p in papers],
            [(e.source_paper_id, e.target_paper_id) for e in edges],
        )

        nodes = []
        for paper in papers:
            payload = self._serialize(paper)
            payload.update(
                {
                    "in_degree": degree_in.get(paper.id, 0),
                    "out_degree": degree_out.get(paper.id, 0),
                    "hub_score": round(hubs.get(paper.id, 0.0), 4),
                    "authority_score": round(authorities.get(paper.id, 0.0), 4),
                }
            )
            nodes.append(payload)

        return {
            "nodes": nodes,
            "edges": [
                {
                    "id": edge.id,
                    "source": edge.source_paper_id,
                    "target": edge.target_paper_id,
                    "relation": edge.relation,
                    "weight": edge.weight,
                }
                for edge in edges
            ],
            "stats": {
                "nodes": len(nodes),
                "edges": len(edges),
                "density": round(len(edges) / max(len(nodes) * (len(nodes) - 1), 1), 6),
            },
        }

    def tag_stats(self) -> list[dict]:
        rows = self.session.execute(
            select(PaperTag.tag, PaperTag.category, func.count(PaperTag.id))
            .group_by(PaperTag.tag, PaperTag.category)
            .order_by(func.count(PaperTag.id).desc())
        ).all()
        return [{"tag": tag, "category": category, "count": count} for tag, category, count in rows]

    def library_stats(self) -> dict:
        total = self.session.scalar(select(func.count()).select_from(Paper)) or 0
        edges = self.session.scalar(select(func.count()).select_from(Citation)) or 0
        by_source = self.session.execute(
            select(Paper.source, func.count(Paper.id)).group_by(Paper.source)
        ).all()
        years = self.session.execute(
            select(Paper.year, func.count(Paper.id))
            .where(Paper.year.is_not(None))
            .group_by(Paper.year)
            .order_by(Paper.year)
        ).all()
        return {
            "papers": total,
            "citations": edges,
            "by_source": [{"source": s, "count": c} for s, c in by_source],
            "by_year": [{"year": y, "count": c} for y, c in years],
            "tags": self.tag_stats(),
            "taxonomy": taxonomy_catalog(),
        }

    async def provider_health(self) -> list[dict]:
        providers = list(provider_registry().values())
        health = await asyncio.gather(*(p.health() for p in providers))
        return [{**provider.metadata(), **status} for provider, status in zip(providers, health, strict=True)]

    def hits_report(self, limit: int = 10) -> dict:
        provider = get_provider("hits_mcp")
        assert isinstance(provider, HITSMCPResearchProvider)
        return provider.hub_authority_report(limit=limit)

    def provider_log(self, limit: int = 50) -> list[dict]:
        rows = self.session.scalars(
            select(ProviderQuery).order_by(ProviderQuery.created_at.desc()).limit(limit)
        ).all()
        return [
            {
                "id": row.id,
                "provider": row.provider,
                "query": row.query,
                "result_count": row.result_count,
                "degraded": row.degraded,
                "latency_ms": row.latency_ms,
                "error": row.error,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]

    # ------------------------------------------------------------ internals
    @staticmethod
    def _merge(responses: list[ProviderResponse]) -> list[PaperRecord]:
        """Deduplicate across providers by normalized title, keeping the richest record."""
        merged: dict[str, PaperRecord] = {}
        for response in responses:
            for record in response.papers:
                key = record.dedupe_key
                if not key:
                    continue
                existing = merged.get(key)
                if existing is None:
                    merged[key] = record
                    continue
                # Prefer whichever record carries more metadata.
                if len(record.abstract) > len(existing.abstract):
                    existing.abstract = record.abstract
                existing.citation_count = max(existing.citation_count, record.citation_count)
                existing.relevance = max(existing.relevance, record.relevance)
                existing.url = existing.url or record.url
                existing.pdf_url = existing.pdf_url or record.pdf_url
                existing.venue = existing.venue or record.venue
                existing.year = existing.year or record.year
                existing.authors = existing.authors or record.authors
                existing.tags = sorted(set(existing.tags) | set(record.tags))
                existing.references = sorted(set(existing.references) | set(record.references))
        return list(merged.values())

    def _find_by_title(self, title: str) -> Paper | None:
        normalized = normalize_title(title)
        for paper in self.session.scalars(select(Paper)).unique():
            if normalize_title(paper.title) == normalized:
                return paper
        return None

    def _upsert(self, record: PaperRecord) -> Paper:
        paper = self.session.scalar(
            select(Paper).where(
                Paper.source == record.source, Paper.external_id == record.external_id
            )
        )
        if paper is None:
            paper = self._find_by_title(record.title)

        if paper is None:
            paper = Paper(
                external_id=record.external_id,
                source=record.source,
                title=record.title,
            )
            self.session.add(paper)

        paper.abstract = record.abstract or paper.abstract
        paper.authors = record.authors or paper.authors
        paper.year = record.year or paper.year
        paper.venue = record.venue or paper.venue
        paper.url = record.url or paper.url
        paper.pdf_url = record.pdf_url or paper.pdf_url
        paper.citation_count = max(record.citation_count, paper.citation_count or 0)
        paper.relevance = max(record.relevance, paper.relevance or 0.0)
        paper.notes = record.notes or paper.notes
        paper.discovered_via = record.discovered_via or paper.discovered_via

        self.session.flush()
        self._sync_tags(paper, record.tags)
        return paper

    def _sync_tags(self, paper: Paper, tags: list[str]) -> None:
        existing = {t.tag for t in paper.tags}
        category_of = {c.name: c.name for c in CATEGORIES}
        for tag in tags:
            if tag in existing:
                continue
            self.session.add(
                PaperTag(paper_id=paper.id, tag=tag, category=category_of.get(tag, "General"))
            )

    def _link(self, source_id: str, target_id: str, *, relation: str = "cites", weight: float = 1.0) -> None:
        if source_id == target_id:
            return
        exists = self.session.scalar(
            select(Citation).where(
                Citation.source_paper_id == source_id, Citation.target_paper_id == target_id
            )
        )
        if exists:
            return
        self.session.add(
            Citation(
                source_paper_id=source_id,
                target_paper_id=target_id,
                relation=relation,
                weight=weight,
            )
        )

    def _link_reference_titles(self, records: list[PaperRecord], papers: list[Paper]) -> None:
        """Resolve provider-supplied reference strings against the library."""
        key_index = corpus_key_index()
        by_record = {record.dedupe_key: paper for record, paper in zip(records, papers, strict=True)}
        for record in records:
            source = by_record.get(record.dedupe_key)
            if source is None:
                continue
            for reference in record.references:
                title = key_index[reference].title if reference in key_index else reference
                target = self._find_by_title(title)
                if target is not None:
                    self._link(source.id, target.id)

    def _recompute_offline_citation_counts(self) -> None:
        """Papers with no provider-supplied count get their in-corpus in-degree instead."""
        counts: dict[str, int] = defaultdict(int)
        for edge in self.session.scalars(select(Citation)).unique():
            counts[edge.target_paper_id] += 1
        for paper in self.session.scalars(select(Paper)).unique():
            if paper.citation_count == 0 and counts.get(paper.id):
                paper.citation_count = counts[paper.id]

    def _neighbors(self, paper_id: str, *, outgoing: bool) -> list[Paper]:
        column = Citation.source_paper_id if outgoing else Citation.target_paper_id
        other = Citation.target_paper_id if outgoing else Citation.source_paper_id
        ids = self.session.scalars(select(other).where(column == paper_id)).all()
        if not ids:
            return []
        return list(self.session.scalars(select(Paper).where(Paper.id.in_(ids))).unique())

    def _audit(self, response: ProviderResponse) -> None:
        self.session.add(
            ProviderQuery(
                provider=response.provider,
                query=response.query,
                result_count=len(response.papers),
                degraded=response.degraded,
                latency_ms=response.latency_ms,
                error=response.error[:1000],
            )
        )

    @staticmethod
    def _serialize(paper: Paper) -> dict:
        return {
            "id": paper.id,
            "external_id": paper.external_id,
            "source": paper.source,
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": list(paper.authors or []),
            "year": paper.year,
            "venue": paper.venue,
            "url": paper.url,
            "pdf_url": paper.pdf_url,
            "citation_count": paper.citation_count,
            "relevance": paper.relevance,
            "notes": paper.notes,
            "discovered_via": paper.discovered_via,
            "tags": sorted(t.tag for t in paper.tags),
        }
