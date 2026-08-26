from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("api") / "test.db"
    import app.db as db_module
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False})
    db_module.engine = engine
    db_module.SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    db_module.Base.metadata.create_all(engine)

    with TestClient(create_app()) as client:
        yield client


def test_meta_exposes_agents_actions_and_policies(client):
    payload = client.get("/api/meta").json()
    assert {a["id"] for a in payload["agents"]} == {
        "generalist",
        "planner",
        "researcher",
        "critic",
        "verifier",
        "memory",
        "executor",
    }
    assert len(payload["actions"]) == 10
    assert {p["id"] for p in payload["policies"]} >= {
        "contextual_bandit",
        "mdp",
        "markov_game",
        "marl",
    }
    assert len(payload["taxonomy"]) == 9


def test_run_lifecycle(client):
    created = client.post(
        "/api/runs",
        json={
            "task": "Survey reward shaping for cooperative multi-agent orchestration",
            "policy": "markov_game",
            "seed": 42,
            "max_steps": 12,
        },
    )
    assert created.status_code == 201
    run = created.json()
    run_id = run["id"]
    assert run["step_count"] == 0
    assert run["preview"]["distribution"]

    stepped = client.post(f"/api/runs/{run_id}/step", json={"steps": 5}).json()
    assert len(stepped["steps"]) >= 1
    first = stepped["steps"][0]
    assert first["entropy_before"] - first["entropy_after"] == pytest.approx(
        first["information_gain"], abs=1e-9
    )

    traces = client.get(f"/api/runs/{run_id}/traces").json()
    assert len(traces) == len(stepped["steps"])
    assert traces[0]["action_distribution"]

    metrics = client.get(f"/api/runs/{run_id}/metrics").json()
    assert metrics["totals"]["steps"] == len(traces)
    assert metrics["series"]

    graph = client.get(f"/api/runs/{run_id}/graph").json()
    assert graph["edges"]

    reset = client.post(f"/api/runs/{run_id}/reset", json={"seed": 7}).json()
    assert reset["step_count"] == 0
    assert client.get(f"/api/runs/{run_id}/traces").json() == []

    assert client.delete(f"/api/runs/{run_id}").status_code == 204
    assert client.get(f"/api/runs/{run_id}").status_code == 404


def _budget_run(**overrides) -> dict:
    payload = {
        "task": "Find why p99 latency tripled after a logging-only deploy",
        "strategy": "control",
        "seed": 5,
        "max_steps": 2,
        "mode": "live",
        "belief_dim": 4,
        "hypotheses": ["h0", "h1", "h2", "h3"],
    }
    payload.update(overrides)
    return payload


def test_budget_ceiling_scales_with_the_cost_unit(client):
    # 250,000 is absurd in dollars and ordinary in tokens, so the bound cannot be a constant.
    in_tokens = client.post(
        "/api/runs", json=_budget_run(cost_unit="tokens", budget_usd=250_000)
    )
    assert in_tokens.status_code == 201
    client.delete(f"/api/runs/{in_tokens.json()['id']}")

    in_dollars = client.post("/api/runs", json=_budget_run(budget_usd=250_000))
    assert in_dollars.status_code == 422
    assert "ceiling" in in_dollars.text


def test_a_dollar_budget_left_on_a_token_run_is_refused(client):
    # Silently accepting 1.20 tokens produces a zero-step run that looks like a result.
    response = client.post("/api/runs", json=_budget_run(cost_unit="tokens"))
    assert response.status_code == 422
    assert "too small to be denominated" in response.text


def test_research_library_is_seeded_and_queryable(client):
    stats = client.get("/api/research/stats").json()
    assert stats["papers"] >= 30
    assert stats["citations"] >= 30

    papers = client.get("/api/research/papers", params={"tag": "MARL", "limit": 5}).json()
    assert papers
    detail = client.get(f"/api/research/papers/{papers[0]['id']}").json()
    assert "references" in detail and "cited_by" in detail

    graph = client.get("/api/research/graph").json()
    assert graph["stats"]["edges"] > 0

    providers = client.get("/api/research/providers").json()
    assert {p["id"] for p in providers} >= {
        "arxiv",
        "semantic_scholar",
        "papers_with_code",
        "hits_mcp",
        "local_corpus",
    }

    hits = client.get("/api/research/hits").json()
    assert hits["authorities"]


def test_search_persists_into_the_library(client):
    response = client.post(
        "/api/research/search",
        json={"query": "value decomposition cooperative agents", "providers": ["local_corpus"], "limit": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] > 0
    assert payload["providers"][0]["provider"] == "local_corpus"

    log = client.get("/api/research/log").json()
    assert log[0]["provider"] == "local_corpus"


def test_websocket_streams_steps(client):
    run_id = client.post(
        "/api/runs",
        json={"task": "Stream a short episode", "policy": "mdp", "seed": 5, "max_steps": 6},
    ).json()["id"]

    with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot"
        ws.send_json({"type": "step"})
        event = ws.receive_json()
        assert event["type"] == "step"
        assert event["step"]["step"] == 1
        ws.send_json({"type": "close"})
