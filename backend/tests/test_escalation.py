from __future__ import annotations

import numpy as np
import pytest

from app.orchestration.actions import ACTIONS, Action, SPECIALIST_ACTIONS
from app.orchestration.agents import AGENTS, SOLO_AGENT
from app.orchestration.engine import OrchestrationEngine, RunConfig
from app.orchestration.state import FEATURE_NAMES, initial_state
from app.orchestration.transitions import TransitionModel


def build(policy: str = "heuristic", seed: int = 7, **overrides) -> OrchestrationEngine:
    return OrchestrationEngine(
        RunConfig(task="escalation", policy=policy, seed=seed, max_steps=30, **overrides)
    )


def run(engine: OrchestrationEngine) -> list[Action]:
    taken = []
    while not engine.done:
        taken.append(Action(engine.step().action))
    return taken


# --------------------------------------------------------------- action space


def test_solo_is_the_only_agent_before_escalation():
    engine = build()
    legal = engine.legal_actions()
    assert Action.INVOKE_GENERALIST in legal
    for action in SPECIALIST_ACTIONS:
        assert action not in legal


def test_escalation_requires_a_solo_attempt_first():
    """You have to try the task before you are allowed to declare it too big."""
    engine = build()
    assert Action.ESCALATE not in engine.legal_actions()

    engine.step()
    assert Action.ESCALATE in engine.legal_actions()


def test_specialists_unlock_only_after_escalating():
    engine = build(policy="fixed_sequence", budget_usd=20.0)
    engine.step()
    assert not engine.state.has_escalated

    engine.step()
    assert engine.state.has_escalated
    legal = engine.legal_actions()
    assert Action.INVOKE_PLANNER in legal
    assert Action.INVOKE_GENERALIST not in legal


def test_escalation_is_irreversible():
    engine = build(policy="fixed_sequence", budget_usd=20.0)
    while not engine.state.has_escalated and not engine.done:
        engine.step()
    for _ in range(4):
        if engine.done:
            break
        engine.step()
        assert engine.state.has_escalated


def test_min_solo_steps_is_configurable():
    engine = build(policy="fixed_sequence", budget_usd=20.0, min_solo_steps=3)
    engine.step()
    assert Action.ESCALATE not in engine.legal_actions()
    engine.step()
    engine.step()
    assert Action.ESCALATE in engine.legal_actions()


# ------------------------------------------------------------------ mechanics


def test_escalation_costs_money_but_resolves_nothing():
    engine = build(policy="fixed_sequence", budget_usd=20.0, escalation_cost_usd=0.07)
    engine.step()
    before = engine.state
    unresolved_before = before.unresolved_subtasks

    step = engine.step()
    assert step.action == Action.ESCALATE.value
    assert step.cost_usd == pytest.approx(0.07)
    assert step.tokens == 0
    assert step.information_gain == pytest.approx(0.0)
    assert engine.state.unresolved_subtasks == unresolved_before


def test_escalation_is_a_pure_cost_in_the_reward():
    engine = build(policy="fixed_sequence", budget_usd=20.0)
    engine.step()
    step = engine.step()

    assert step.action == Action.ESCALATE.value
    assert step.reward < 0.0
    assert step.reward_breakdown["quality"] == pytest.approx(0.0)
    assert step.reward_breakdown["progress"] == pytest.approx(0.0)


def test_generalist_is_excluded_from_coalitions():
    engine = build(policy="markov_game", budget_usd=20.0)
    for _ in range(25):
        if engine.done:
            break
        step = engine.step()
        if step.action == Action.RUN_PARALLEL.value:
            assert SOLO_AGENT not in step.agents


# ------------------------------------------------------------------- baselines


def test_never_escalate_control_stays_solo():
    engine = build(policy="single_agent", budget_usd=20.0)
    actions = run(engine)
    assert Action.ESCALATE not in actions
    assert engine.state.has_escalated is False


def test_always_escalate_bookend_escalates_immediately():
    engine = build(policy="fixed_sequence", budget_usd=20.0)
    engine.step()
    engine.step()
    assert engine.state.has_escalated


def test_allow_escalation_false_removes_the_option_entirely():
    engine = build(policy="fixed_sequence", budget_usd=20.0, allow_escalation=False)
    actions = run(engine)
    assert Action.ESCALATE not in actions
    assert set(actions) <= {Action.INVOKE_GENERALIST, Action.TERMINATE}


# ----------------------------------------------------------------- stall signal


def test_stall_is_exposed_to_the_policy():
    assert "stall" in FEATURE_NAMES
    assert "has_escalated" in FEATURE_NAMES


def test_stall_rises_when_nothing_moves():
    state = initial_state(
        task_complexity=0.5,
        budget_usd=5.0,
        latency_budget_ms=90_000.0,
        belief_dim=4,
        rng=np.random.default_rng(2),
    )
    assert state.stall == 0.0

    state.stall_steps = 3
    assert state.stall == pytest.approx(1.0)
    state.stall_steps = 9
    assert state.stall == pytest.approx(1.0), "stall saturates rather than growing without bound"


def test_escalation_clears_the_stall():
    engine = build(policy="fixed_sequence", budget_usd=20.0)
    engine.step()
    engine.state.stall_steps = 3
    engine.step()
    assert engine.state.stall_steps == 0


# ------------------------------------------------------ duplicate penalty fix


def test_productive_repeats_are_not_punished():
    """The old term charged for any repeat within four steps, which taught policies to quit."""
    model = TransitionModel(belief_dim=4)
    state = initial_state(
        task_complexity=0.5,
        budget_usd=5.0,
        latency_budget_ms=90_000.0,
        belief_dim=4,
        rng=np.random.default_rng(3),
    )
    state.invocation_signatures = ["researcher", "researcher", "researcher"]
    state.last_agents = ["researcher"]

    productive, _ = model._duplicate_pressure(state, ["researcher"], information_gain=0.5)
    barren, _ = model._duplicate_pressure(state, ["researcher"], information_gain=0.0)

    assert productive == pytest.approx(0.0)
    assert barren > 0.5


def test_repetition_still_costs_when_it_yields_nothing():
    model = TransitionModel(belief_dim=4)
    state = initial_state(
        task_complexity=0.5,
        budget_usd=5.0,
        latency_budget_ms=90_000.0,
        belief_dim=4,
        rng=np.random.default_rng(4),
    )
    state.last_agents = ["critic"]

    once, _ = model._duplicate_pressure(state, ["critic"], information_gain=0.0)
    state.invocation_signatures = ["critic", "critic"]
    thrice, _ = model._duplicate_pressure(state, ["critic"], information_gain=0.0)

    assert thrice > once


def test_action_space_has_both_control_actions():
    assert Action.ESCALATE in ACTIONS
    assert Action.TERMINATE in ACTIONS
    assert AGENTS[SOLO_AGENT].id == "generalist"
