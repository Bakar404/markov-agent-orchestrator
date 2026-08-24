from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.orchestration.policies import POLICY_REGISTRY
from app.orchestration.strategies import STRATEGIES, get_strategy
from app.research.taxonomy import taxonomy_catalog


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("catalog") / "test.db"
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


# ------------------------------------------------------------ strategy catalog


def test_every_strategy_names_a_real_policy():
    for strategy in STRATEGIES:
        assert strategy.policy in POLICY_REGISTRY, strategy.id


def test_every_strategy_names_a_real_taxonomy_category():
    """The library is the arm catalog, so a broken link would orphan the papers."""
    known = {c["name"] for c in taxonomy_catalog()}
    for strategy in STRATEGIES:
        assert strategy.category in known, f"{strategy.id} -> {strategy.category}"


def test_exactly_one_control_strategy():
    controls = [s for s in STRATEGIES if s.is_control]
    assert len(controls) == 1
    assert controls[0].id == "control"
    assert controls[0].escalates == "never"


def test_escalation_bookends_exist():
    """Never and always bracket the gate; without both there is nothing to compare against."""
    modes = {s.escalates for s in STRATEGIES}
    assert {"never", "always"} <= modes


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        get_strategy("nope")


def test_meta_exposes_strategies(client):
    payload = client.get("/api/meta").json()
    assert {s["id"] for s in payload["strategies"]} == {s.id for s in STRATEGIES}


# ------------------------------------------------------- creating from strategy


def test_run_can_be_created_from_a_strategy(client):
    response = client.post(
        "/api/runs",
        json={"task": "compare approaches", "strategy": "control", "max_steps": 4},
    )
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["policy"] == "single_agent"
    assert run["config"]["arm"] == "control"
    assert run["config"]["policy_options"]["agent_id"] == "generalist"


def test_explicit_arm_overrides_the_strategy_name(client):
    run = client.post(
        "/api/runs",
        json={
            "task": "compare approaches",
            "strategy": "always_orchestrate",
            "arm": "my-pipeline",
            "max_steps": 4,
        },
    ).json()
    assert run["config"]["arm"] == "my-pipeline"
    assert run["policy"] == "fixed_sequence"


def test_unknown_strategy_is_a_400(client):
    response = client.post(
        "/api/runs", json={"task": "compare approaches", "strategy": "not-real"}
    )
    assert response.status_code == 400


# -------------------------------------------------------------------- verdicts


def make_arm(client, *, experiment: str, strategy: str, seed: int) -> str:
    run = client.post(
        "/api/runs",
        json={
            "task": "Which caching strategy fits a read-heavy API",
            "strategy": strategy,
            "seed": seed,
            "max_steps": 6,
            "experiment": experiment,
        },
    ).json()
    client.post(f"/api/runs/{run['id']}/step", json={"steps": 6})
    return run["id"]


def test_verdict_is_recorded_and_replaces_on_rejudge(client):
    run_id = make_arm(client, experiment="verdicts", strategy="control", seed=1)

    first = client.post(f"/api/runs/{run_id}/verdict", json={"score": 0.4, "rubric": "v1"})
    assert first.status_code == 201

    second = client.post(f"/api/runs/{run_id}/verdict", json={"score": 0.9, "rubric": "v2"})
    assert second.status_code == 201
    assert second.json()["score"] == 0.9

    arm = client.get("/api/experiments/verdicts").json()["arms"][0]
    assert arm["mean_quality"] == 0.9
    assert arm["judged_runs"] == 1


def test_verdict_rejects_out_of_range_scores(client):
    run_id = make_arm(client, experiment="range", strategy="control", seed=2)
    assert client.post(f"/api/runs/{run_id}/verdict", json={"score": 1.5}).status_code == 422


def test_verdict_on_missing_run_is_404(client):
    assert client.post("/api/runs/nope/verdict", json={"score": 0.5}).status_code == 404


# ------------------------------------------------------------ paired statistics


def test_deltas_are_paired_on_shared_seeds(client):
    for seed in (11, 12, 13):
        make_arm(client, experiment="paired", strategy="control", seed=seed)
        make_arm(client, experiment="paired", strategy="always_orchestrate", seed=seed)

    payload = client.get("/api/experiments/paired").json()
    arms = {a["arm"]: a for a in payload["arms"]}

    delta = arms["always_orchestrate"]["vs_control"]
    assert delta["paired_seeds"] == 3
    assert delta["cost_usd"]["n"] == 3
    assert "mean_delta" in delta["cost_usd"]
    assert "stderr" in delta["cost_usd"]


def test_unshared_seeds_are_reported_rather_than_silently_averaged(client):
    make_arm(client, experiment="unpaired", strategy="control", seed=21)
    make_arm(client, experiment="unpaired", strategy="always_orchestrate", seed=99)

    arms = {a["arm"]: a for a in client.get("/api/experiments/unpaired").json()["arms"]}
    delta = arms["always_orchestrate"]["vs_control"]
    assert delta["paired_seeds"] == 0
    assert "same seeds" in delta["note"]


def test_comparison_never_ranks_on_internal_reward(client):
    make_arm(client, experiment="honest", strategy="control", seed=31)
    payload = client.get("/api/experiments/honest").json()

    for arm in payload["arms"]:
        assert "mean_internal_reward" in arm
        if arm["vs_control"]:
            assert "internal_reward" not in arm["vs_control"]
    assert any("never compared" in c for c in payload["caveats"])


def test_verdict_refuses_without_quality_judgments(client):
    make_arm(client, experiment="noquality", strategy="control", seed=41)
    make_arm(client, experiment="noquality", strategy="always_orchestrate", seed=41)

    payload = client.get("/api/experiments/noquality").json()
    assert "No verdict" in payload["verdict"]
    assert any("Unjudged arms" in c for c in payload["caveats"])


def test_missing_control_is_called_out(client):
    make_arm(client, experiment="nocontrol", strategy="always_orchestrate", seed=51)
    payload = client.get("/api/experiments/nocontrol").json()

    assert payload["control_arm"] is None
    assert any("no arm is named" in c.lower() for c in payload["caveats"])


def test_thin_evidence_is_flagged(client):
    make_arm(client, experiment="thin", strategy="control", seed=61)
    payload = client.get("/api/experiments/thin").json()
    assert any("Thin evidence" in c for c in payload["caveats"])
