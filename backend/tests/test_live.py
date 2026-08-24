from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.orchestration.agents import AGENTS
from app.orchestration.live import report_from_response


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("live") / "test.db"
    import app.db as db_module
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False}
    )
    db_module.engine = engine
    db_module.SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )
    db_module.Base.metadata.create_all(engine)

    with TestClient(create_app()) as client:
        yield client


def create_live_run(client, **overrides) -> str:
    body = {
        "task": "Determine the best policy family for cooperative orchestration",
        "policy": "mdp",
        "seed": 11,
        "max_steps": 12,
        "belief_dim": 4,
        "mode": "live",
    }
    body.update(overrides)
    response = client.post("/api/runs", json=body)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def reports_for(pending, *, outcome="success", confidence=0.8, hypothesis=1):
    return [
        {
            "agent_id": agent_id,
            "outcome": outcome,
            "confidence": confidence,
            "claimed_hypothesis": hypothesis,
            "response": f"{agent_id} produced a real answer.",
            "tokens": 1234,
            "latency_ms": 900.0,
            "cost_usd": 0.02,
        }
        for agent_id in pending["agents"]
    ]


def test_open_returns_a_brief_without_advancing(client):
    run_id = create_live_run(client)

    pending = client.post(f"/api/runs/{run_id}/live/open").json()
    assert pending["token"]
    assert pending["step"] == 1
    assert pending["action_distribution"]

    run = client.get(f"/api/runs/{run_id}").json()
    assert run["step_count"] == 0, "opening a step must not advance the episode"

    if pending["agents"]:
        brief = pending["briefs"][0]
        assert brief["instruction"]
        assert brief["context"]["task"]
        assert len(brief["hypotheses"]) == 4


def test_report_advances_and_records_measured_cost(client):
    run_id = create_live_run(client, seed=12)
    pending = client.post(f"/api/runs/{run_id}/live/open").json()
    if not pending["agents"]:
        pytest.skip("policy opened with TERMINATE")

    response = client.post(
        f"/api/runs/{run_id}/live/report",
        json={"token": pending["token"], "reports": reports_for(pending)},
    )
    assert response.status_code == 200, response.text
    step = response.json()["step"]

    assert step["step"] == 1
    assert step["cost_usd"] == pytest.approx(0.02 * len(pending["agents"]))
    assert step["tokens"] == 1234 * len(pending["agents"])
    assert all(report["source"] == "live" for report in step["reports"])
    assert all(report["claimed_hypothesis"] == 1 for report in step["reports"])


def test_belief_mass_follows_the_claim(client):
    run_id = create_live_run(client, seed=13)
    pending = client.post(f"/api/runs/{run_id}/live/open").json()
    if not pending["agents"]:
        pytest.skip("policy opened with TERMINATE")

    before = client.get(f"/api/runs/{run_id}").json()["state"]["belief"][2]
    client.post(
        f"/api/runs/{run_id}/live/report",
        json={"token": pending["token"], "reports": reports_for(pending, hypothesis=2)},
    )
    after = client.get(f"/api/runs/{run_id}").json()["state"]["belief"][2]
    assert after > before


def test_stale_token_is_rejected(client):
    run_id = create_live_run(client, seed=14)
    pending = client.post(f"/api/runs/{run_id}/live/open").json()
    if not pending["agents"]:
        pytest.skip("policy opened with TERMINATE")

    response = client.post(
        f"/api/runs/{run_id}/live/report",
        json={"token": "not-the-open-token", "reports": reports_for(pending)},
    )
    assert response.status_code == 409


def test_report_without_open_is_rejected(client):
    run_id = create_live_run(client, seed=15)
    response = client.post(
        f"/api/runs/{run_id}/live/report",
        json={"token": "anything", "reports": []},
    )
    assert response.status_code == 409


def test_wrong_agent_set_is_rejected(client):
    run_id = create_live_run(client, seed=16)
    pending = client.post(f"/api/runs/{run_id}/live/open").json()
    if not pending["agents"]:
        pytest.skip("policy opened with TERMINATE")

    bad = reports_for(pending)
    bad[0]["agent_id"] = "planner" if pending["agents"][0] != "planner" else "critic"
    response = client.post(
        f"/api/runs/{run_id}/live/report",
        json={"token": pending["token"], "reports": bad},
    )
    assert response.status_code == 422


def test_unknown_agent_id_is_rejected(client):
    run_id = create_live_run(client, seed=17)
    pending = client.post(f"/api/runs/{run_id}/live/open").json()
    if not pending["agents"]:
        pytest.skip("policy opened with TERMINATE")

    bad = reports_for(pending)
    bad[0]["agent_id"] = "nonexistent"
    response = client.post(
        f"/api/runs/{run_id}/live/report",
        json={"token": pending["token"], "reports": bad},
    )
    assert response.status_code == 422


def test_disagreement_raises_entropy_more_than_agreement():
    """The core live-mode claim: with no oracle, confidence may only come from agreement.

    Driven through the kernel directly so it does not depend on the policy electing a coalition.
    """
    import numpy as np

    from app.orchestration.actions import Action
    from app.orchestration.state import initial_state
    from app.orchestration.transitions import TransitionModel

    def entropy_after(claims: list[int]) -> float:
        rng = np.random.default_rng(5)
        state = initial_state(
            task_complexity=0.5,
            budget_usd=2.0,
            latency_budget_ms=90_000.0,
            belief_dim=4,
            rng=rng,
        )
        agents = ["researcher", "verifier"]
        reports = [
            report_from_response(
                AGENTS[agent_id],
                state,
                {
                    "outcome": "success",
                    "confidence": 0.9,
                    "claimed_hypothesis": claim,
                    "response": "evidence",
                },
                belief_dim=4,
            )
            for agent_id, claim in zip(agents, claims, strict=True)
        ]
        outcome = TransitionModel(belief_dim=4, stochasticity=0.0).sample(
            state,
            Action.RUN_PARALLEL,
            np.random.default_rng(5),
            agent_ids=agents,
            reports_override=reports,
        )
        return outcome.entropy_after

    assert entropy_after([1, 3]) > entropy_after([1, 1])


def test_sim_run_rejects_live_endpoints(client):
    run_id = create_live_run(client, seed=18, mode="sim")
    assert client.post(f"/api/runs/{run_id}/live/open").status_code == 409


def test_hypotheses_can_be_named_on_first_report(client):
    run_id = create_live_run(client, seed=19)
    pending = client.post(f"/api/runs/{run_id}/live/open").json()
    if not pending["agents"]:
        pytest.skip("policy opened with TERMINATE")

    names = ["vdn-mixing", "synergy-matrix", "linucb", "tabular-q"]
    client.post(
        f"/api/runs/{run_id}/live/report",
        json={
            "token": pending["token"],
            "reports": reports_for(pending),
            "hypotheses": names,
        },
    )
    follow_up = client.post(f"/api/runs/{run_id}/live/open").json()
    if follow_up.get("briefs"):
        assert follow_up["briefs"][0]["hypotheses"] == names


def test_wrong_hypothesis_count_is_rejected(client):
    run_id = create_live_run(client, seed=20)
    pending = client.post(f"/api/runs/{run_id}/live/open").json()
    if not pending["agents"]:
        pytest.skip("policy opened with TERMINATE")

    response = client.post(
        f"/api/runs/{run_id}/live/report",
        json={
            "token": pending["token"],
            "reports": reports_for(pending),
            "hypotheses": ["only", "two"],
        },
    )
    assert response.status_code == 422
