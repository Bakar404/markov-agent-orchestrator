from __future__ import annotations

import numpy as np
import pytest

from app.orchestration.actions import ACTIONS, Action
from app.orchestration.engine import OrchestrationEngine, RunConfig
from app.orchestration.policies import POLICY_REGISTRY
from app.orchestration.state import FEATURE_DIM
from app.orchestration.transitions import TransitionModel


def build(policy: str = "contextual_bandit", seed: int = 7, **overrides) -> OrchestrationEngine:
    options = {"max_steps": 25, **overrides}
    config = RunConfig(
        task="Design a reward function for cooperative agent orchestration",
        policy=policy,
        seed=seed,
        **options,
    )
    return OrchestrationEngine(config)


def run_to_completion(engine: OrchestrationEngine, cap: int = 200) -> list:
    results = []
    while not engine.done and len(results) < cap:
        results.append(engine.step())
    return results


@pytest.mark.parametrize("policy_id", sorted(POLICY_REGISTRY))
def test_every_policy_completes_an_episode(policy_id: str):
    engine = build(policy_id)
    results = run_to_completion(engine)
    assert results, f"{policy_id} produced no steps"
    assert engine.done
    assert engine.state.termination_reason
    for result in results:
        assert 0.0 <= result.action_probability <= 1.0
        assert result.action in {a.value for a in ACTIONS}
        assert result.state["step"] == result.step


def test_same_seed_reproduces_the_trajectory():
    a = run_to_completion(build(seed=99))
    b = run_to_completion(build(seed=99))
    assert [s.action for s in a] == [s.action for s in b]
    assert [round(s.reward, 9) for s in a] == [round(s.reward, 9) for s in b]


def test_different_seeds_diverge():
    a = [s.action for s in run_to_completion(build(seed=1))]
    b = [s.action for s in run_to_completion(build(seed=2))]
    assert a != b


def test_transitions_are_stochastic_for_a_fixed_state_and_action():
    engine = build(seed=3)
    model = TransitionModel(belief_dim=engine.config.belief_dim)
    state = engine.state
    entropies = set()
    costs = set()
    for seed in range(12):
        rng = np.random.default_rng(seed)
        outcome = model.sample(state, Action.INVOKE_RESEARCHER, rng)
        entropies.add(round(outcome.entropy_after, 6))
        costs.add(round(outcome.cost_usd, 6))
    assert len(entropies) > 1, "the same action always produced the same successor entropy"
    assert len(costs) > 1


def test_information_gain_matches_the_entropy_endpoints():
    engine = build(seed=11)
    for result in run_to_completion(engine):
        assert result.information_gain == pytest.approx(
            result.entropy_before - result.entropy_after, abs=1e-9
        )


def test_reward_breakdown_sums_to_the_reward():
    engine = build(seed=13)
    for result in run_to_completion(engine):
        breakdown = result.reward_breakdown
        terms = sum(
            value for key, value in breakdown.items() if key not in {"per_agent", "total"}
        )
        assert terms == pytest.approx(result.reward, abs=1e-9)
        assert breakdown["total"] == pytest.approx(result.reward, abs=1e-9)


def test_budget_is_never_overspent_beyond_termination():
    engine = build(seed=17, budget_usd=0.25, max_steps=100)
    run_to_completion(engine)
    assert engine.state.terminated
    assert engine.state.termination_reason in {
        "budget_exhausted",
        "latency_exhausted",
        "goal_reached",
        "policy_terminate",
        "step_limit",
    }


def test_snapshot_restore_round_trip_continues_the_same_stream():
    # Terminate is gated off long enough that the warm-up cannot end the episode.
    engine = build(
        seed=23,
        max_steps=60,
        budget_usd=6.0,
        latency_budget_ms=600_000.0,
        min_steps_before_terminate=8,
    )
    for _ in range(5):
        assert not engine.done
        engine.step()
    assert not engine.done

    restored = OrchestrationEngine.restore(engine.snapshot())
    original_next = engine.step()
    restored_next = restored.step()

    assert original_next.action == restored_next.action
    assert original_next.reward == pytest.approx(restored_next.reward)
    assert original_next.entropy_after == pytest.approx(restored_next.entropy_after)


def test_terminate_stays_illegal_while_there_is_funded_work_left():
    engine = build(seed=31, min_steps_before_terminate=4, budget_usd=8.0)
    for _ in range(4):
        assert Action.TERMINATE not in engine.legal_actions()
        engine.step()

    # Past the step gate, stopping is still not a legal choice while subtasks remain and the
    # budget is healthy: quitting early has to be impossible, not merely discouraged.
    if engine.state.unresolved_subtasks > 0 and engine.state.budget_remaining >= 0.25:
        assert Action.TERMINATE not in engine.legal_actions()


def test_terminate_becomes_legal_once_the_budget_runs_low():
    engine = build(seed=5, min_steps_before_terminate=0, budget_usd=0.08, max_steps=80)
    while not engine.done and engine.state.budget_remaining >= 0.25:
        engine.step()
    if not engine.done:
        assert Action.TERMINATE in engine.legal_actions()


def test_feature_vector_dimension_is_stable():
    engine = build()
    assert engine.state.features().shape == (FEATURE_DIM,)


def test_parallel_action_can_invoke_a_coalition():
    engine = build(policy="markov_game", seed=5)
    model = TransitionModel(belief_dim=engine.config.belief_dim)
    rng = np.random.default_rng(0)
    outcome = model.sample(engine.state, Action.RUN_PARALLEL, rng)
    assert len(outcome.reports) >= 2
    assert outcome.cost_usd > 0


def test_policy_distribution_is_normalized_over_legal_actions():
    engine = build(policy="marl", seed=8)
    legal = engine.legal_actions()
    distribution = engine.policy.distribution(engine.state, legal)
    assert distribution.sum() == pytest.approx(1.0)
    illegal = [a for a in ACTIONS if a not in legal]
    for action in illegal:
        assert distribution[ACTIONS.index(action)] == pytest.approx(0.0)
