from __future__ import annotations

import asyncio
import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Citation, Paper
from app.research.corpus import corpus_edges, corpus_records, search_corpus
from app.research.hits_mcp import HITSMCPResearchProvider, hits_scores
from app.research.service import ResearchService
from app.research.taxonomy import CATEGORY_NAMES, classify, relevance_score


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    with factory() as session:
        yield session


def test_corpus_loads_and_has_citation_edges():
    records = corpus_records()
    assert len(records) >= 30
    assert all(record.title for record in records)
    assert len(corpus_edges()) >= 30


def test_corpus_search_ranks_topical_matches_first():
    results = search_corpus("markov games multi-agent reinforcement learning", limit=5)
    assert results
    titles = " ".join(r.title.lower() for r in results)
    assert "markov" in titles or "multi-agent" in titles


def test_every_taxonomy_category_is_represented():
    covered = set()
    for record in corpus_records():
        covered.update(record.tags)
    missing = set(CATEGORY_NAMES) - covered
    assert not missing, f"uncovered categories: {sorted(missing)}"


def test_classify_assigns_known_categories():
    tags = classify(
        "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning",
        "value decomposition for cooperative multi-agent reinforcement learning",
    )
    assert "MARL" in tags
    assert all(tag in CATEGORY_NAMES for tag in tags)


def test_relevance_score_is_bounded_and_query_sensitive():
    high = relevance_score(
        title="Markov games as a framework for multi-agent reinforcement learning",
        abstract="minimax-Q for zero-sum Markov games",
        query="markov games multi-agent reinforcement learning",
        year=1994,
        citation_count=5000,
        tags=["Markov Games", "MARL", "Stochastic Games"],
    )
    low = relevance_score(
        title="A study of soil composition",
        abstract="agronomy field trials",
        query="markov games multi-agent reinforcement learning",
        year=1994,
        citation_count=0,
        tags=["Planning"],
    )
    assert 0.0 <= low < high <= 1.0


def test_hits_separates_hubs_from_authorities():
    nodes = ["survey", "foundation_a", "foundation_b", "applied"]
    edges = [
        ("survey", "foundation_a"),
        ("survey", "foundation_b"),
        ("applied", "foundation_a"),
    ]
    hubs, authorities = hits_scores(nodes, edges)
    assert hubs["survey"] > hubs["foundation_a"]
    assert authorities["foundation_a"] > authorities["survey"]


def test_seeding_populates_papers_tags_and_citations(session):
    service = ResearchService(session)
    created = service.seed_if_empty()
    assert created >= 30
    assert session.query(Paper).count() == created
    assert session.query(Citation).count() >= 30

    # Idempotent
    assert service.seed_if_empty() == 0

    stats = service.library_stats()
    assert stats["papers"] == created
    assert stats["citations"] >= 30
    assert len(stats["tags"]) > 0


def test_citation_graph_reports_degrees_and_hits(session):
    service = ResearchService(session)
    service.seed_if_empty()
    graph = service.citation_graph(limit=200)
    assert graph["stats"]["nodes"] > 0
    assert graph["stats"]["edges"] > 0
    assert any(node["in_degree"] > 0 for node in graph["nodes"])
    assert any(node["authority_score"] > 0 for node in graph["nodes"])


def test_offline_search_degrades_instead_of_failing(session, monkeypatch):
    monkeypatch.setenv("RESEARCH_ALLOW_NETWORK", "false")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        service = ResearchService(session)
        service.seed_if_empty()
        result = asyncio.run(
            service.search("cooperative markov game orchestration", provider_ids=["arxiv"], limit=5)
        )
        assert result["degraded"] is True
        assert result["count"] > 0
        assert result["providers"][0]["provider"] == "arxiv"
    finally:
        os.environ.pop("RESEARCH_ALLOW_NETWORK", None)
        get_settings.cache_clear()


def test_hits_mcp_provider_falls_back_to_local_ranking():
    provider = HITSMCPResearchProvider()
    response = asyncio.run(provider.search("multi-agent reinforcement learning", limit=6))
    assert response.papers
    assert response.degraded is True
    assert all(p.source == "hits_mcp" for p in response.papers)

    report = provider.hub_authority_report(limit=5)
    assert report["authorities"]
    assert report["hubs"]
