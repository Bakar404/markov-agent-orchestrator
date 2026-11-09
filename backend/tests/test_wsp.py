from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.orchestration import wsp_guard
from app.orchestration.agents import AGENT_IDS
from app.orchestration.policies import create_policy
from app.orchestration.state import FEATURE_DIM, FEATURE_NAMES
from app.orchestration.strategies import get_strategy
from app.orchestration.wsp import Edit, Patch, PatchFormatError
from app.orchestration.wsp.apply import GuardLimits, PatchRejected, apply_patch, edit_metrics
from app.orchestration.wsp.calibrate import (
    CALIBRATION_SEED_BAND,
    SeedLeakError,
    SeedSplit,
    calibrate,
)
from app.orchestration.wsp.editor import RecordedEditor, example_path, parse_reply
from app.orchestration.wsp.program import DEFAULT_GOAL, CheckpointError, load_base
from app.orchestration.wsp.render import render_policy

BAND_SEED = CALIBRATION_SEED_BAND.start + 3


def _stub_check(shapes, current, edits, limits):
    """Minimal limit check standing in for the verified guard while it is a placeholder."""
    total = sum(math.prod(shape) for shape in shapes.values())
    for edit in edits:
        before = float(np.asarray(current[edit.tensor])[tuple(edit.index)])
        if abs(edit.value) > limits.max_abs_value:
            return wsp_guard.GuardVerdict(False, "value too large", "max_abs_value")
        if abs(edit.value - before) > limits.max_abs_delta:
            return wsp_guard.GuardVerdict(False, "delta too large", "max_abs_delta")
    if len(edits) / total > limits.max_changed_fraction:
        return wsp_guard.GuardVerdict(False, "too many edits", "max_changed_fraction")
    return wsp_guard.GuardVerdict(True)


def _guard_is_placeholder() -> bool:
    try:
        wsp_guard.check_patch({"synergy": (1, 1)}, {"synergy": [[0.0]]}, [], GuardLimits())
    except NotImplementedError:
        return True
    return False


@pytest.fixture
def guard(monkeypatch):
    if _guard_is_placeholder():
        monkeypatch.setattr(wsp_guard, "check_patch", _stub_check)


@pytest.fixture(autouse=True)
def no_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("WSP_CHECKPOINT", str(tmp_path / "absent.json"))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("wsp") / "test.db"
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.db as db_module

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


def example() -> dict:
    return json.loads(example_path("narrow_coalitions").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ patch contract
def test_patch_round_trips_through_json():
    patch = Patch.from_dict(example())
    again = Patch.from_json(patch.to_json())
    assert again == patch
    payload = json.loads(patch.to_json())
    assert set(payload) == {"goal", "rationale", "edits"}
    assert all(set(e) == {"tensor", "index", "value"} for e in payload["edits"])
    assert payload == example()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("rationale"),
        lambda p: p.update(extra=1),
        lambda p: p["edits"][0].update(tensor="bias"),
        lambda p: p["edits"][0].update(index=[6, "0"]),
        lambda p: p["edits"][0].update(index=[True, 0]),
        lambda p: p["edits"][0].update(value="3"),
        lambda p: p["edits"][0].update(value=float("nan")),
        lambda p: p["edits"][0].update(delta=1.0),
    ],
)
def test_malformed_patches_are_format_errors(mutate):
    payload = example()
    mutate(payload)
    with pytest.raises(PatchFormatError):
        Patch.from_dict(payload)


# ------------------------------------------------------------------ renderer
def test_renderer_labels_every_axis_and_shows_only_the_upper_triangle():
    policy = create_policy("markov_game", feature_dim=FEATURE_DIM)
    policy.synergy[1, 3] = 0.5
    policy.synergy[3, 1] = 9.99  # never read by the policy, so it must not be shown
    text = render_policy(policy, limits=GuardLimits())

    for agent in AGENT_IDS:
        assert f"| {agent} |" in text
    for j, name in enumerate(FEATURE_NAMES):
        assert f"{j}:{name}" in text
        assert f"| [{j}] | {name} |" in text

    n = len(AGENT_IDS)
    for i in range(n):
        for j in range(n):
            row = f"| [{i}, {j}] | {AGENT_IDS[i]} + {AGENT_IDS[j]} |"
            assert (row in text) == (i < j)
    assert "| [1, 3] | researcher + verifier | +0.500 |" in text
    assert "9.990" not in text
    assert "≤ 9 entries" in text


# ------------------------------------------------------------------ seed split
def test_seed_split_is_enforced():
    SeedSplit.build(evaluation=[101, 102])
    with pytest.raises(SeedLeakError):
        SeedSplit.build(evaluation=[BAND_SEED])
    with pytest.raises(SeedLeakError):
        SeedSplit(calibration=(101,))
    with pytest.raises(SeedLeakError):
        SeedSplit(calibration=(BAND_SEED,), evaluation=(BAND_SEED,))
    with pytest.raises(SeedLeakError):
        calibrate(create_policy("markov_game", feature_dim=FEATURE_DIM).state_dict(), [101])


def test_evaluation_runs_cannot_use_calibration_seeds(client, guard):
    for body in (
        {"strategy": "wsp_markov_game"},
        {"strategy": "control", "experiment": "leak"},
    ):
        response = client.post(
            "/api/runs", json={"task": "seed leak", "seed": BAND_SEED, **body}
        )
        assert response.status_code == 400, body
        assert "reserved" in response.json()["detail"]


def test_random_seeds_skip_the_calibration_band(monkeypatch):
    from app.services import run_service

    draws = iter([BAND_SEED, 77])
    monkeypatch.setattr(run_service.random, "randrange", lambda *_: next(draws))
    assert run_service.draw_seed() == 77


# ------------------------------------------------------------------ apply
def test_apply_refuses_a_patch_the_guard_rejects(monkeypatch):
    policy = create_policy("markov_game", feature_dim=FEATURE_DIM)
    seen = {}

    def reject(shapes, current, edits, limits):
        seen.update(shapes=shapes, edits=list(edits), limits=limits)
        return wsp_guard.GuardVerdict(False, "nope", "max_abs_delta")

    monkeypatch.setattr(wsp_guard, "check_patch", reject)
    patch = Patch.from_dict(example())
    with pytest.raises(PatchRejected) as info:
        apply_patch(policy, patch, GuardLimits())
    assert info.value.verdict.rule == "max_abs_delta"
    assert seen["shapes"] == {"q_weights": (7, 17), "synergy": (7, 7), "terminate_weights": (17,)}
    assert seen["edits"] == list(patch.edits)
    assert not policy.q_weights.any() and not policy.synergy.any()


def test_apply_fails_closed_when_the_guard_is_unavailable(monkeypatch):
    def unavailable(*_):
        raise NotImplementedError

    monkeypatch.setattr(wsp_guard, "check_patch", unavailable)
    with pytest.raises(PatchRejected) as info:
        apply_patch(create_policy("markov_game", feature_dim=FEATURE_DIM),
                    Patch.from_dict(example()))
    assert info.value.verdict.rule == "guard_unavailable"


def test_apply_refuses_dead_synergy_entries_even_if_the_guard_accepts(monkeypatch):
    monkeypatch.setattr(wsp_guard, "check_patch", lambda *_: wsp_guard.GuardVerdict(True))
    patch = Patch(goal="g", rationale="r", edits=(Edit("synergy", (3, 1), 0.5),))
    with pytest.raises(PatchRejected):
        apply_patch(create_policy("markov_game", feature_dim=FEATURE_DIM), patch)


def test_edit_metrics_on_a_hand_built_example(monkeypatch):
    monkeypatch.setattr(wsp_guard, "check_patch", lambda *_: wsp_guard.GuardVerdict(True))
    policy = SimpleNamespace(
        q_weights=np.array([[3.0, 4.0], [0.0, 0.0]]),
        synergy=np.zeros((2, 2)),
        terminate_weights=np.array([1.0, 1.0]),
    )
    patch = Patch(goal="g", rationale="r", edits=(
        Edit("q_weights", (0, 0), 0.0),
        Edit("synergy", (0, 1), 2.0),
        Edit("terminate_weights", (1,), 1.0),  # same value: not a change
    ))
    applied = apply_patch(policy, patch)
    m = applied.metrics

    assert m["total_parameters"] == 10
    assert m["effective_parameters"] == 7
    assert m["changed_entries"] == 2
    assert m["changed_fraction"] == pytest.approx(2 / 7)
    assert m["changed_fraction_raw"] == pytest.approx(0.2)
    assert m["changed_pct"] == pytest.approx(100 * 2 / 7)
    q, s, t = (m["per_tensor"][k] for k in ("q_weights", "synergy", "terminate_weights"))
    assert (q["changed"], q["dw_fro"], q["w_fro"], q["relative_dw_fro"]) == (1, 3.0, 5.0, 0.6)
    assert s["dw_fro"] == 2.0 and s["relative_dw_fro"] is None
    assert t["changed"] == 0 and t["relative_dw_fro"] == 0.0
    assert m["dw_fro"] == pytest.approx(math.sqrt(13))
    assert m["relative_dw_fro"] == pytest.approx(math.sqrt(13) / math.sqrt(27))
    top = m["top_changed"][0]
    assert top["label"] == "q_weights[planner, bias]"
    assert (top["before"], top["after"], top["delta"]) == (3.0, 0.0, -3.0)
    # The caller's tensors are untouched; the patched copy carries the edits.
    assert policy.q_weights[0, 0] == 3.0 and applied.policy.q_weights[0, 0] == 0.0
    assert edit_metrics({"q_weights": np.ones(1), "synergy": np.ones(1),
                         "terminate_weights": np.ones(1)},
                        {"q_weights": np.ones(1), "synergy": np.ones(1),
                         "terminate_weights": np.ones(1)})["changed_entries"] == 0


# ------------------------------------------------------------------ editors
def test_recorded_editor_refuses_a_patch_for_another_goal():
    editor = RecordedEditor.example("narrow_coalitions")
    assert editor.propose("", "  " + DEFAULT_GOAL.replace(" ", "  ")) == Patch.from_dict(example())
    with pytest.raises(PatchFormatError):
        editor.propose("", "Always escalate")
    with pytest.raises(PatchFormatError):
        RecordedEditor.example("../../secrets")


def test_copilot_reply_parsing_is_offline_and_strict():
    payload = json.dumps(example())
    expected = Patch.from_dict(example())
    assert parse_reply(payload) == expected
    assert parse_reply(f"\n```json\n{payload}\n```\n") == expected
    refused = [
        "I would rather not.",
        f"Sure. {payload} Done.",  # prose around the object is not searched through
        f"Here you go:\n```json\n{payload}\n```",
        f"{payload}\n{payload}",
        payload[:-1] + ', "goal": "something else"}',  # a repeated key
        payload.replace('"value": 3.0', '"value": NaN'),
        payload[:-1] + ', "limits": {"max_abs_value": 1e9}}',  # cannot widen the guard
        json.dumps({**example(), "rationale": "x" * 70_000}),
        "[" * 5000 + "]" * 5000,
        '{"goal": ' + "[" * 50_000,  # nested past the recursion limit
    ]
    for reply in refused:
        with pytest.raises(PatchFormatError):
            parse_reply(reply)


def test_default_goal_matches_the_strategy_and_the_shipped_patch():
    assert get_strategy("wsp_markov_game").programming["goal"] == DEFAULT_GOAL
    assert example()["goal"] == DEFAULT_GOAL


@pytest.mark.skipif(_guard_is_placeholder(), reason="wsp_guard is still the placeholder")
def test_shipped_patch_passes_the_real_guard():
    policy = create_policy("markov_game", feature_dim=FEATURE_DIM)
    applied = apply_patch(policy, Patch.from_dict(example()), GuardLimits())
    assert applied.verdict.accepted


# ------------------------------------------------------------------ checkpoint
def test_base_comes_from_a_checkpoint_when_one_exists(monkeypatch, tmp_path):
    state = create_policy("markov_game", feature_dim=FEATURE_DIM).state_dict()
    state["q_weights"][2][5] = 1.25
    path = tmp_path / "markov_game.json"
    path.write_text(json.dumps({"policy_state": state}), encoding="utf-8")
    monkeypatch.setenv("WSP_CHECKPOINT", str(path))
    policy, base = load_base()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert base == {"source": "checkpoint", "checkpoint": "markov_game.json", "sha256": digest}
    assert policy.q_weights[2, 5] == 1.25

    state["synergy"] = [[0.0]]
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(CheckpointError) as caught:
        load_base()
    # The 400 this becomes must not disclose where the server keeps its checkpoints.
    assert str(tmp_path) not in str(caught.value)

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckpointError) as caught:
        load_base()
    assert str(tmp_path) not in str(caught.value)


# ------------------------------------------------------------------ end to end
def test_wsp_strategy_is_in_the_catalog(client):
    strategies = {s["id"]: s for s in client.get("/api/meta").json()["strategies"]}
    wsp = strategies["wsp_markov_game"]
    assert wsp["policy"] == "markov_game"
    assert wsp["programming"]["method"] == "wsp"
    assert strategies["control"]["programming"] == {}


def test_wsp_arm_runs_end_to_end_offline(client, guard):
    from app.services.run_service import _engines

    created = client.post(
        "/api/runs",
        json={"task": "pick a cache", "strategy": "wsp_markov_game", "seed": 101,
              "experiment": "wsp-e2e", "max_steps": 12},
    )
    assert created.status_code == 201, created.text
    run = created.json()
    assert run["config"]["arm"] == "wsp_markov_game"
    assert run["wsp"]["accepted"] and run["wsp"]["editor"] == "recorded"
    assert run["wsp"]["changed_entries"] == 8
    # Effective = 56 q_weights + 15 synergy pairs among the six non-generalist agents + 80
    # terminate weights = 151; the generalist's six pairs are never read by any coalition.
    assert run["wsp"]["changed_pct"] == pytest.approx(100 * 8 / 151)
    assert run["wsp"]["changed_fraction"] == pytest.approx(8 / 151)
    assert run["wsp"]["changed_fraction_raw"] == pytest.approx(8 / 185)

    policy = _engines[run["id"]].policy
    assert policy.q_weights[6, 0] == 3.0 and policy.synergy[1, 3] == 0.2

    artifact = client.get(f"/api/runs/{run['id']}/wsp").json()
    assert artifact["patch"] == Patch.from_dict(example()).to_dict()
    assert artifact["verdict"]["accepted"] is True
    assert artifact["metrics"]["effective_parameters"] == 151
    assert artifact["metrics"]["changed_fraction_raw"] == pytest.approx(8 / 185)
    assert artifact["base"]["source"] == "default_init"
    assert all(s in CALIBRATION_SEED_BAND for s in artifact["calibration_seeds"])
    assert 101 not in artifact["calibration_seeds"]
    assert "reward" not in json.dumps(artifact["calibration"])

    stepped = client.post(f"/api/runs/{run['id']}/step", json={"steps": 50}).json()
    assert stepped["run"]["terminated"]

    reset = client.post(f"/api/runs/{run['id']}/reset", json={"seed": 102})
    assert reset.status_code == 200
    assert _engines[run["id"]].policy.q_weights[6, 0] == 3.0
    assert client.post(f"/api/runs/{run['id']}/reset", json={"seed": BAND_SEED}).status_code == 400

    control = client.post(
        "/api/runs",
        json={"task": "pick a cache", "strategy": "control", "seed": 101, "experiment": "wsp-e2e"},
    ).json()
    assert control["wsp"] is None
    assert client.get(f"/api/runs/{control['id']}/wsp").status_code == 404


def _provenance(client, **params) -> dict:
    brief = client.get("/api/wsp/brief", params=params).json()
    return {"base_sha256": brief["base"]["sha256"],
            "calibration_seeds": brief["calibration"]["seeds"]}


def test_rejected_patch_creates_no_run(client, guard):
    before = len(client.get("/api/runs").json())
    bad = {**example(), "edits": [{"tensor": "q_weights", "index": [6, 0], "value": 50.0}]}
    response = client.post(
        "/api/runs",
        json={"task": "pick a cache", "strategy": "wsp_markov_game", "seed": 5, "wsp_patch": bad,
              "wsp_provenance": _provenance(client)},
    )
    assert response.status_code == 400
    assert "rejected" in response.json()["detail"]
    assert len(client.get("/api/runs").json()) == before


def test_inline_patch_needs_and_records_provenance(client, guard):
    body = {"task": "pick a cache", "strategy": "wsp_markov_game", "seed": 7,
            "wsp_patch": example()}
    missing = client.post("/api/runs", json=body)
    assert missing.status_code == 400 and "wsp_provenance" in missing.json()["detail"]

    wrong = {"base_sha256": "0" * 64, "calibration_seeds": _provenance(client)["calibration_seeds"]}
    stale = client.post("/api/runs", json={**body, "wsp_provenance": wrong})
    assert stale.status_code == 400 and "base checkpoint changed" in stale.json()["detail"]

    provenance = _provenance(client)
    created = client.post("/api/runs", json={**body, "wsp_provenance": provenance})
    assert created.status_code == 201, created.text
    artifact = client.get(f"/api/runs/{created.json()['id']}/wsp").json()
    assert artifact["base"]["patch_provenance"] == provenance
    assert artifact["base"]["sha256"] == provenance["base_sha256"]
    assert artifact["calibration_seeds"] == provenance["calibration_seeds"]


def test_experiment_refuses_a_base_that_changed_between_seeds(client, guard, monkeypatch, tmp_path):
    body = {"task": "pick a cache", "strategy": "wsp_markov_game", "experiment": "pinned"}
    first = client.post("/api/runs", json={**body, "seed": 11})
    assert first.status_code == 201, first.text

    state = create_policy("markov_game", feature_dim=FEATURE_DIM).state_dict()
    state["q_weights"][0][0] = 0.5
    path = tmp_path / "markov_game.json"
    path.write_text(json.dumps({"policy_state": state}), encoding="utf-8")
    monkeypatch.setenv("WSP_CHECKPOINT", str(path))
    second = client.post("/api/runs", json={**body, "seed": 12})
    assert second.status_code == 400
    assert "base checkpoint changed" in second.json()["detail"]
    # Another experiment is free to use the new base.
    other = client.post("/api/runs", json={**body, "experiment": "fresh", "seed": 12})
    assert other.status_code == 201, other.text


def test_programmed_weights_keep_the_runs_policy_options(client, guard, monkeypatch, tmp_path):
    from app.services.run_service import _engines

    # A checkpoint saved with different hyperparameters must not override the run's own.
    policy = create_policy("markov_game", feature_dim=FEATURE_DIM, temperature=3.0,
                           learning_rate=0.5, discount=0.1)
    path = tmp_path / "markov_game.json"
    path.write_text(json.dumps({"policy_state": policy.state_dict()}), encoding="utf-8")
    monkeypatch.setenv("WSP_CHECKPOINT", str(path))
    options = {"temperature": 0.9, "learning_rate": 0.01, "discount": 0.5}
    created = client.post(
        "/api/runs",
        json={"task": "pick a cache", "strategy": "wsp_markov_game", "seed": 21,
              "policy_options": options},
    )
    assert created.status_code == 201, created.text
    run = created.json()
    stored = run["config"]["policy_options"]
    for engine in (_engines[run["id"]],):
        for name, value in options.items():
            assert stored[name] == value
            assert getattr(engine.policy, name) == pytest.approx(value)
    assert client.post(f"/api/runs/{run['id']}/reset", json={"seed": 22}).status_code == 200
    for name, value in options.items():
        assert getattr(_engines[run["id"]].policy, name) == pytest.approx(value)


def test_calibration_runs_under_the_evaluation_task_config(client, guard):
    created = client.post(
        "/api/runs",
        json={"task": "pick a cache", "strategy": "wsp_markov_game", "seed": 31,
              "budget_usd": 3.5, "latency_budget_ms": 40_000, "task_complexity": 0.8,
              "task_shape": {"needs_evidence": 0.9}},
    )
    assert created.status_code == 201, created.text
    calibration = client.get(f"/api/runs/{created.json()['id']}/wsp").json()["calibration"]
    overrides = calibration["overrides"]
    assert overrides["budget_usd"] == 3.5
    assert overrides["latency_budget_ms"] == 40_000
    assert overrides["task_complexity"] == 0.8
    assert overrides["task_shape"]["needs_evidence"] == 0.9


def test_calibration_is_cached_per_base_and_task_config(client, monkeypatch):
    from app.orchestration.wsp import program as program_module

    calls = []
    real = program_module.calibrate

    def counting(*args, **kwargs):
        calls.append(kwargs.get("budget_usd"))
        return real(*args, **kwargs)

    monkeypatch.setattr(program_module, "calibrate", counting)
    program_module._CALIBRATION_CACHE.clear()
    first = client.get("/api/wsp/brief", params={"max_steps": 9}).json()
    again = client.get("/api/wsp/brief", params={"max_steps": 9}).json()
    assert len(calls) == 1 and first["calibration"] == again["calibration"]
    client.get("/api/wsp/brief", params={"max_steps": 9, "budget_usd": 2.0})
    assert len(calls) == 2


def test_wsp_fields_need_a_programmed_strategy(client):
    response = client.post(
        "/api/runs", json={"task": "pick a cache", "strategy": "control", "wsp_goal": "anything"}
    )
    assert response.status_code == 400


def test_brief_shows_rendered_weights_and_calibration(client):
    payload = client.get("/api/wsp/brief", params={"max_steps": 10}).json()
    assert payload["default_goal"] == DEFAULT_GOAL
    assert "## synergy" in payload["rendered"]
    assert all(s in CALIBRATION_SEED_BAND for s in payload["calibration"]["seeds"])
    assert payload["limits"]["max_changed_fraction"] == 0.05


# ------------------------------------------------------------------ synergy is upper-triangle
def test_reversed_agent_order_reads_and_writes_the_edited_upper_synergy():
    from app.orchestration.actions import Action
    from app.orchestration.state import initial_state

    policy = create_policy("markov_game", feature_dim=FEATURE_DIM)
    i, j = 1, 3
    policy.synergy[i, j] = 0.7  # what an accepted patch writes; the lower entry is left stale
    policy.synergy[j, i] = 0.0
    state = initial_state(task_complexity=0.5, budget_usd=2.0, latency_budget_ms=90_000.0,
                          belief_dim=4, rng=np.random.default_rng(3))
    forward = policy.coalition_value(state, (i, j))
    assert policy.coalition_value(state, (j, i)) == pytest.approx(forward)

    # A live report listing the agents in reverse order must learn exactly what the forward
    # order learns: on the edited upper entry, never from the stale lower one.
    import copy

    reverse = copy.deepcopy(policy)
    forward_agents = [AGENT_IDS[i], AGENT_IDS[j]]
    reverse_agents = forward_agents[::-1]
    policy.update(state, Action.RUN_PARALLEL, forward_agents, 1.0,
                  {a: 0.5 for a in forward_agents}, state, True)
    reverse.update(state, Action.RUN_PARALLEL, reverse_agents, 1.0,
                   {a: 0.5 for a in reverse_agents}, state, True)
    np.testing.assert_allclose(reverse.synergy, policy.synergy)
    np.testing.assert_allclose(reverse.q_weights, policy.q_weights)
    assert reverse.synergy[i, j] == reverse.synergy[j, i]
    assert reverse.synergy[i, j] != pytest.approx(0.0, abs=0.3), "the edit was overwritten"
