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


def drive_live(
    client,
    *,
    arm: str,
    seed: int,
    experiment: str,
    summary: str,
    metered: bool = True,
    steps: int = 2,
):
    """Drive a live run. Steps differ from each other; seeds repeat the same sequence."""
    body = {
        "task": "does adding agents help",
        "strategy": arm,
        "arm": arm,
        "seed": seed,
        "experiment": experiment,
        "mode": "live",
        "max_steps": steps,
        "budget_usd": 5.0,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]

    last = None
    for step in range(steps):
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


def test_an_arm_cut_short_is_not_silently_compared(client):
    """A run stopped by an outage looks cheap for a reason unrelated to how it orchestrates."""
    drive_live(client, arm="control", seed=401, experiment="uneven", summary="full run", steps=5)
    drive_live(
        client,
        arm="always_orchestrate",
        seed=401,
        experiment="uneven",
        summary="cut short by an outage",
        steps=1,
    )

    payload = client.get("/api/experiments/uneven").json()
    assert payload["lopsided_seeds"] == [401]
    assert any("UNEVEN RUNS" in c for c in payload["caveats"])


def test_arms_of_similar_length_are_not_flagged(client):
    drive_live(client, arm="control", seed=402, experiment="even", summary="a full run", steps=5)
    drive_live(
        client, arm="always_orchestrate", seed=402, experiment="even", summary="also full", steps=5
    )

    payload = client.get("/api/experiments/even").json()
    assert payload["lopsided_seeds"] == []
    assert not any("UNEVEN RUNS" in c for c in payload["caveats"])


def test_a_decisive_control_win_is_reported_even_when_another_arm_did_better(client):
    """A challenger that loses every comparison is as informative as one that wins every one."""
    for seed in range(601, 606):
        control, _ = drive_live(
            client, arm="control", seed=seed, experiment="swept", summary=f"solo {seed}"
        )
        swept, _ = drive_live(
            client,
            arm="always_orchestrate",
            seed=seed,
            experiment="swept",
            summary=f"swept {seed}",
        )
        close, _ = drive_live(
            client, arm="cascade", seed=seed, experiment="swept", summary=f"close {seed}"
        )
        # always_orchestrate loses every pair; cascade splits, so it has the better win rate
        # and would previously have been the only arm the headline looked at.
        client.post(
            "/api/experiments/swept/pairwise",
            json={"run_a": control, "run_b": swept, "winner": "a", "judge": "test"},
        )
        client.post(
            "/api/experiments/swept/pairwise",
            json={
                "run_a": control,
                "run_b": close,
                "winner": "a" if seed % 2 else "b",
                "judge": "test",
            },
        )

    verdict = client.get("/api/experiments/swept").json()["verdict"]
    assert "The control won" in verdict, verdict
    assert "always_orchestrate" in verdict, verdict


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


# ------------------------------------------------- externally driven arms


def _external_run(client, *, seed: int, experiment: str) -> str:
    return client.post(
        "/api/runs",
        json={
            "task": "who should act next",
            "strategy": "maf_sequential",
            "arm": "maf_sequential",
            "seed": seed,
            "experiment": experiment,
            "mode": "live",
            "max_steps": 8,
            "budget_usd": 5.0,
        },
    ).json()["id"]


def _declare(client, run_id: str, action: str, agents: list[str]):
    return client.post(
        f"/api/runs/{run_id}/live/open", json={"action": action, "agents": agents}
    )


def _report(client, run_id: str, opened: dict, note: str):
    reports = [
        {
            "agent_id": agent_id,
            "outcome": "success",
            "confidence": 0.7,
            "summary": f"{note} by {agent_id}",
            "response": f"{note} by {agent_id}",
            "tokens": 900,
            "latency_ms": 50.0,
            "cost_usd": 0.02,
        }
        for agent_id in opened["agents"]
    ]
    return client.post(
        f"/api/runs/{run_id}/live/report",
        json={"token": opened["token"], "reports": reports},
    )


def _walk_to_escalation(client, run_id: str) -> None:
    """Do the solo work the arena requires before specialists become reachable."""
    for attempt in range(6):
        legal = client.get(f"/api/runs/{run_id}").json()["preview"]["legal_actions"]
        if "escalate" in legal:
            opened = _declare(client, run_id, "escalate", []).json()
            _report(client, run_id, opened, "escalating")
            return
        opened = _declare(client, run_id, "invoke_generalist", ["generalist"]).json()
        _report(client, run_id, opened, f"solo attempt {attempt}")
    raise AssertionError("never reached escalation")


def test_an_external_run_will_not_open_without_a_declaration(client):
    """Nothing chose, so there is no step to hand back."""
    run_id = _external_run(client, seed=701, experiment="external")
    response = client.post(f"/api/runs/{run_id}/live/open")
    assert response.status_code == 409, response.text
    assert "externally driven" in response.text


def test_an_external_run_takes_the_declared_agents(client):
    run_id = _external_run(client, seed=702, experiment="external")
    _walk_to_escalation(client, run_id)

    opened = _declare(client, run_id, "run_parallel", ["researcher", "critic"])
    assert opened.status_code == 200, opened.text
    body = opened.json()
    assert body["agents"] == ["researcher", "critic"]
    # Nothing was sampled, so claiming a probability below 1 would be fiction.
    assert body["action_probability"] == 1.0


def test_an_external_run_still_obeys_the_escalation_gate(client):
    """Deferring the choice is not the same as exempting it from the rules."""
    run_id = _external_run(client, seed=703, experiment="external")
    response = _declare(client, run_id, "invoke_researcher", ["researcher"])
    assert response.status_code == 409, response.text
    assert "not legal" in response.text


def test_a_policy_driven_run_refuses_a_declaration(client):
    """Otherwise any arm could be steered by its driver and still be labelled a policy."""
    run_id = client.post(
        "/api/runs",
        json={
            "task": "who chooses",
            "strategy": "cascade",
            "arm": "cascade",
            "seed": 704,
            "experiment": "external",
            "mode": "live",
            "max_steps": 4,
        },
    ).json()["id"]

    response = _declare(client, run_id, "invoke_generalist", ["generalist"])
    assert response.status_code == 409, response.text
    assert "chooses for itself" in response.text


def test_a_declaration_drives_exactly_one_step(client):
    """A stale choice must not silently decide a later step it was not made for."""
    run_id = _external_run(client, seed=705, experiment="external")
    opened = _declare(client, run_id, "invoke_generalist", ["generalist"]).json()
    assert _report(client, run_id, opened, "the first solo attempt").status_code == 200

    response = client.post(f"/api/runs/{run_id}/live/open")
    assert response.status_code == 409, response.text
    assert "externally driven" in response.text


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
