from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.orchestration.actions import Action, SINGLE_AGENT_ACTIONS
from app.orchestration.engine import OrchestrationEngine, RunConfig
from app.orchestration.policies import POLICY_REGISTRY, create_policy
from app.orchestration.state import FEATURE_DIM


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("experiments") / "test.db"
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


# ------------------------------------------------------------ control policy


def test_single_agent_is_registered():
    assert "single_agent" in POLICY_REGISTRY


def test_control_invokes_only_its_agent():
    """The never-escalate control can only ever reach the generalist."""
    engine = OrchestrationEngine(
        RunConfig(
            task="control",
            policy="single_agent",
            seed=4,
            budget_usd=20.0,
            max_steps=10,
        )
    )
    invoked = set()
    while not engine.done:
        step = engine.step()
        invoked.update(step.agents)

    assert invoked <= {"generalist"}
    assert engine.state.has_escalated is False


def test_control_never_fans_out():
    engine = OrchestrationEngine(
        RunConfig(task="control", policy="single_agent", seed=6, budget_usd=20.0, max_steps=10)
    )
    while not engine.done:
        assert Action(engine.step().action) is not Action.RUN_PARALLEL


def test_control_rejects_an_unknown_agent():
    with pytest.raises(ValueError):
        create_policy("single_agent", feature_dim=FEATURE_DIM, agent_id="nobody")


def test_control_agent_survives_a_snapshot_round_trip():
    engine = OrchestrationEngine(
        RunConfig(
            task="control",
            policy="single_agent",
            seed=8,
            budget_usd=20.0,
            max_steps=10,
            policy_options={"agent_id": "critic"},
        )
    )
    engine.step()
    restored = OrchestrationEngine.restore(engine.snapshot())
    assert restored.policy.agent_id == "critic"


def test_control_defaults_to_a_valid_agent():
    policy = create_policy("single_agent", feature_dim=FEATURE_DIM)
    assert policy.agent_id in set(SINGLE_AGENT_ACTIONS.values())


# ---------------------------------------------------------------- experiment


def make_run(client, *, arm: str, policy: str, experiment: str, **extra) -> str:
    body = {
        "task": "Which caching strategy fits a read-heavy API",
        "policy": policy,
        "seed": 21,
        "max_steps": 8,
        "budget_usd": 1.2,
        "experiment": experiment,
        "arm": arm,
    }
    body.update(extra)
    response = client.post("/api/runs", json=body)
    assert response.status_code == 201, response.text
    run_id = response.json()["id"]
    client.post(f"/api/runs/{run_id}/step", json={"steps": 8})
    return run_id


def test_untagged_runs_are_excluded(client):
    client.post("/api/runs", json={"task": "untagged run", "policy": "random", "max_steps": 3})
    assert client.get("/api/experiments").json() == []


def test_compare_groups_arms_and_reports_multiples(client):
    make_run(client, arm="control", policy="single_agent", experiment="cache-study")
    make_run(client, arm="marl", policy="marl", experiment="cache-study")

    payload = client.get("/api/experiments/cache-study").json()
    arms = {a["arm"]: a for a in payload["arms"]}

    assert set(arms) == {"control", "marl"}
    assert arms["control"]["vs_control"] is None

    delta = arms["marl"]["vs_control"]
    assert delta["paired_seeds"] == 1, "both arms use seed 21, so they must pair"
    assert delta["cost_usd"]["multiple"] is not None


def test_comparison_refuses_to_rank_on_internal_reward(client):
    """Reward presumes decomposition, so ranking on it would let the framework win by default."""
    make_run(client, arm="control", policy="single_agent", experiment="honesty")
    payload = client.get("/api/experiments/honesty").json()

    assert any("never compared" in c for c in payload["caveats"])
    for arm in payload["arms"]:
        assert "mean_internal_reward" in arm
        delta = arm["vs_control"]
        if delta is not None:
            assert "internal_reward" not in " ".join(delta.keys())


# --------------------------------------------------- guards against bad input


def drive_live(client, *, arm: str, seed: int, experiment: str, summary: str, metered: bool = True):
    """Drive a two-step live run. Steps differ from each other; seeds repeat the same pair."""
    body = {
        "task": "does adding agents help",
        "strategy": arm,
        "arm": arm,
        "seed": seed,
        "experiment": experiment,
        "mode": "live",
        "max_steps": 2,
        "budget_usd": 5.0,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]

    last = None
    for step in range(2):
        opened = client.post(f"/api/runs/{run_id}/live/open").json()
        reports = []
        for agent_id in opened["agents"]:
            report = {
                "agent_id": agent_id,
                "outcome": "success",
                "confidence": 0.7,
                "summary": f"{summary} [step {step}]",
                "response": f"{summary} [step {step}]",
            }
            if metered:
                report.update({"tokens": 900 + step, "latency_ms": 500.0, "cost_usd": 0.02})
            reports.append(report)
        last = client.post(
            f"/api/runs/{run_id}/live/report",
            json={"token": opened["token"], "reports": reports},
        )
    return run_id, last


def test_identical_reports_across_seeds_are_flagged(client):
    """Replaying one answer across seeds is not three samples, and must not read as three."""
    for seed in (101, 102, 103):
        drive_live(client, arm="control", seed=seed, experiment="replayed", summary="same answer")

    payload = client.get("/api/experiments/replayed").json()
    arm = next(a for a in payload["arms"] if a["arm"] == "control")

    assert arm["duplicate_runs"] == 2, "three identical runs means two are copies"
    assert any("DUPLICATE WORK" in c for c in payload["caveats"])


def test_distinct_reports_are_not_flagged(client):
    for seed in (201, 202, 203):
        drive_live(
            client,
            arm="control",
            seed=seed,
            experiment="distinct",
            summary=f"a genuinely different answer for {seed}",
        )

    payload = client.get("/api/experiments/distinct").json()
    arm = next(a for a in payload["arms"] if a["arm"] == "control")

    assert arm["duplicate_runs"] == 0
    assert not any("DUPLICATE WORK" in c for c in payload["caveats"])


def test_pairwise_judging_counts_as_judged(client):
    """Pairwise is the preferred method here, so using it must not read as skipping judgement."""
    control, _ = drive_live(
        client, arm="control", seed=301, experiment="judged", summary="the solo answer"
    )
    rival, _ = drive_live(
        client,
        arm="always_orchestrate",
        seed=301,
        experiment="judged",
        summary="the orchestrated answer",
    )

    before = client.get("/api/experiments/judged").json()["caveats"]
    assert any("Unjudged arms" in c for c in before)

    recorded = client.post(
        "/api/experiments/judged/pairwise",
        json={"run_a": control, "run_b": rival, "winner": "b", "judge": "test"},
    )
    assert recorded.status_code == 201

    after = client.get("/api/experiments/judged").json()["caveats"]
    assert not any("Unjudged arms" in c for c in after)


def test_live_report_requires_measured_cost_and_tokens(client):
    """Omitted figures used to be invented from the agent spec. Now they are refused."""
    _, response = drive_live(
        client, arm="control", seed=301, experiment="unmetered", summary="no cost", metered=False
    )
    assert response.status_code == 422, response.text
    body = response.text.lower()
    assert "tokens" in body and "cost_usd" in body


def test_live_report_requires_a_non_empty_response(client):
    run_id = client.post(
        "/api/runs",
        json={
            "task": "empty response",
            "strategy": "control",
            "arm": "control",
            "seed": 501,
            "experiment": "blank",
            "mode": "live",
            "max_steps": 2,
        },
    ).json()["id"]
    opened = client.post(f"/api/runs/{run_id}/live/open").json()

    response = client.post(
        f"/api/runs/{run_id}/live/report",
        json={
            "token": opened["token"],
            "reports": [
                {
                    "agent_id": opened["agents"][0],
                    "outcome": "success",
                    "response": "   ",
                    "tokens": 100,
                    "latency_ms": 10.0,
                    "cost_usd": 0.01,
                }
            ],
        },
    )
    assert response.status_code == 422, response.text


def test_repeating_earlier_work_in_the_same_run_is_refused(client):
    """A second step that resubmits the first step's output did no new work."""
    run_id = client.post(
        "/api/runs",
        json={
            "task": "replay within a run",
            "strategy": "control",
            "arm": "control",
            "seed": 601,
            "experiment": "intra-replay",
            "mode": "live",
            "max_steps": 3,
        },
    ).json()["id"]

    def send(text: str):
        opened = client.post(f"/api/runs/{run_id}/live/open").json()
        return client.post(
            f"/api/runs/{run_id}/live/report",
            json={
                "token": opened["token"],
                "reports": [
                    {
                        "agent_id": opened["agents"][0],
                        "outcome": "success",
                        "summary": text,
                        "response": text,
                        "tokens": 500,
                        "latency_ms": 10.0,
                        "cost_usd": 0.01,
                    }
                ],
            },
        )

    assert send("the first real finding").status_code == 200
    replayed = send("the first real finding")
    assert replayed.status_code == 422, replayed.text
    assert "already recorded" in replayed.text


def test_experiment_listing_counts_arms(client):
    make_run(client, arm="control", policy="single_agent", experiment="listing")
    make_run(client, arm="bandit", policy="contextual_bandit", experiment="listing")

    entry = next(e for e in client.get("/api/experiments").json() if e["experiment"] == "listing")
    assert entry["arms"] == ["bandit", "control"]
    assert entry["runs"] == 2
    assert entry["has_control"] is True


def test_unknown_experiment_is_404(client):
    assert client.get("/api/experiments/never-ran").status_code == 404


def test_arms_record_measurable_quantities(client):
    make_run(client, arm="control", policy="single_agent", experiment="measured")
    arm = client.get("/api/experiments/measured").json()["arms"][0]

    for key in ("mean_cost_usd", "mean_latency_ms", "mean_tokens", "mean_steps"):
        assert isinstance(arm[key], (int, float))
    assert arm["run_ids"]
