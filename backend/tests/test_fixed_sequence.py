from __future__ import annotations

import numpy as np
import pytest

from app.orchestration.actions import Action, SINGLE_AGENT_ACTIONS
from app.orchestration.engine import OrchestrationEngine, RunConfig
from app.orchestration.policies import POLICY_REGISTRY, create_policy
from app.orchestration.policies.fixed_sequence import DEFAULT_SEQUENCE, FixedSequencePolicy
from app.orchestration.state import FEATURE_DIM, initial_state


def build(seed: int = 7, **overrides) -> OrchestrationEngine:
    return OrchestrationEngine(
        RunConfig(task="control arm", policy="fixed_sequence", seed=seed, **overrides)
    )


def a_state():
    return initial_state(
        task_complexity=0.5,
        budget_usd=1.0,
        latency_budget_ms=60_000.0,
        belief_dim=4,
        rng=np.random.default_rng(1),
    )


def test_registered_in_the_catalog():
    assert "fixed_sequence" in POLICY_REGISTRY
    policy = create_policy("fixed_sequence", feature_dim=FEATURE_DIM)
    assert isinstance(policy, FixedSequencePolicy)


def test_walks_the_pipeline_in_order():
    engine = build(budget_usd=8.0, max_steps=len(DEFAULT_SEQUENCE) + 2)
    taken = []
    while not engine.done:
        taken.append(Action(engine.step().action))

    # Escalation is gated behind a solo attempt, so the generalist always acts first.
    assert taken[0] is Action.INVOKE_GENERALIST
    assert taken[1] is Action.ESCALATE

    specialists = [a for a in taken if a in SINGLE_AGENT_ACTIONS and a is not Action.INVOKE_GENERALIST]
    assert specialists == list(DEFAULT_SEQUENCE[: len(specialists)])


def test_cycles_back_to_the_start():
    engine = build(budget_usd=20.0, max_steps=len(DEFAULT_SEQUENCE) * 2 + 2)
    taken = []
    while not engine.done:
        taken.append(Action(engine.step().action))

    specialists = [a for a in taken if a in SINGLE_AGENT_ACTIONS and a is not Action.INVOKE_GENERALIST]
    expected = list(DEFAULT_SEQUENCE) + list(DEFAULT_SEQUENCE)
    assert specialists == expected[: len(specialists)]


def test_never_fans_out_to_a_coalition():
    """Choosing a coalition size is a decision, and the control makes none."""
    engine = build(budget_usd=20.0, max_steps=40)
    while not engine.done:
        assert Action(engine.step().action) is not Action.RUN_PARALLEL


def test_learns_nothing_from_reward():
    policy = create_policy("fixed_sequence", feature_dim=FEATURE_DIM)
    state = a_state()
    before = policy.score_actions(state).copy()

    policy.update(state, Action.TERMINATE, [], 99.0, {}, state, False)
    np.testing.assert_allclose(policy.score_actions(state), before)


def test_identical_routing_regardless_of_seed():
    """No adaptation means the arm sequence must not depend on the RNG stream."""
    a = [s.action for s in _run(build(seed=1, budget_usd=20.0, max_steps=12))]
    b = [s.action for s in _run(build(seed=999, budget_usd=20.0, max_steps=12))]
    assert a == b


def test_task_shape_does_not_change_routing():
    """The control must ignore context; that is what makes it a control."""
    heavy = build(seed=3, budget_usd=20.0, max_steps=12, task_shape={"needs_evidence": 0.95})
    light = build(seed=3, budget_usd=20.0, max_steps=12, task_shape={"needs_evidence": 0.05})
    assert [s.action for s in _run(heavy)] == [s.action for s in _run(light)]


def test_cursor_survives_a_snapshot_round_trip():
    engine = build(budget_usd=20.0, max_steps=12)
    engine.step()
    engine.step()

    restored = OrchestrationEngine.restore(engine.snapshot())
    assert restored.policy.cursor == engine.policy.cursor
    assert restored.policy.diagnostics(restored.state) == engine.policy.diagnostics(engine.state)


def test_diagnostics_name_the_next_stage():
    policy = create_policy("fixed_sequence", feature_dim=FEATURE_DIM)
    diag = policy.diagnostics(a_state())
    assert diag["next_in_pipeline"] == DEFAULT_SEQUENCE[0].value
    assert diag["cycle_length"] == len(DEFAULT_SEQUENCE)
    assert diag["cycles_completed"] == 0


def test_completes_an_episode(monkeypatch):
    engine = build(budget_usd=1.2, max_steps=40)
    while not engine.done:
        engine.step()
    assert engine.state.terminated
    assert engine.state.termination_reason is not None
    assert engine.state.step > 0


def _run(engine: OrchestrationEngine):
    steps = []
    while not engine.done:
        steps.append(engine.step())
    return steps
