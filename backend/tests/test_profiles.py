from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base
from app.orchestration.engine import OrchestrationEngine, RunConfig
from app.orchestration.state import FEATURE_DIM, FEATURE_NAMES, initial_state
from app.services.policy_profile_service import (
    PolicyProfileService,
    ProfileSignatureMismatch,
    signature_for,
)


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'profiles.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    with factory() as s:
        yield s


def build(policy: str = "contextual_bandit", seed: int = 7, **overrides) -> OrchestrationEngine:
    config = RunConfig(
        task="Route this",
        policy=policy,
        seed=seed,
        max_steps=12,
        **overrides,
    )
    return OrchestrationEngine(config)


def run_to_completion(engine: OrchestrationEngine) -> float:
    while not engine.done:
        engine.step()
    return engine.cumulative_reward


# --------------------------------------------------------------- task shape


def test_task_shape_is_part_of_the_context_vector():
    assert set(FEATURE_NAMES) >= {
        "needs_evidence",
        "needs_execution",
        "needs_verification",
        "has_escalated",
        "stall",
    }
    assert FEATURE_DIM == 17


def test_differently_shaped_tasks_are_distinguishable_at_step_zero():
    """The gap this closes: without shape, every task looked identical on the first decision."""
    rng = np.random.default_rng(3)
    common = dict(
        task_complexity=0.5, budget_usd=1.0, latency_budget_ms=60_000.0, belief_dim=4
    )
    research = initial_state(
        **common,
        rng=np.random.default_rng(3),
        task_shape={"needs_evidence": 0.9, "needs_execution": 0.1, "needs_verification": 0.8},
    )
    chore = initial_state(
        **common,
        rng=np.random.default_rng(3),
        task_shape={"needs_evidence": 0.1, "needs_execution": 0.9, "needs_verification": 0.2},
    )
    assert not np.allclose(research.features(), chore.features())
    assert rng is not None


def test_task_shape_stays_out_of_the_tabular_key():
    """Including it would multiply an already-sparse Q-table key space by 8 for no gain."""
    common = dict(
        task_complexity=0.5, budget_usd=1.0, latency_budget_ms=60_000.0, belief_dim=4
    )
    a = initial_state(
        **common, rng=np.random.default_rng(3), task_shape={"needs_evidence": 0.9}
    )
    b = initial_state(
        **common, rng=np.random.default_rng(3), task_shape={"needs_evidence": 0.1}
    )
    assert a.discretize() == b.discretize()
    assert len(a.discretize()) == FEATURE_DIM - 1 - 4


def test_task_shape_defaults_to_neutral_and_survives_a_round_trip():
    state = initial_state(
        task_complexity=0.5,
        budget_usd=1.0,
        latency_budget_ms=60_000.0,
        belief_dim=4,
        rng=np.random.default_rng(1),
    )
    assert state.needs_evidence == 0.5

    shaped = initial_state(
        task_complexity=0.5,
        budget_usd=1.0,
        latency_budget_ms=60_000.0,
        belief_dim=4,
        rng=np.random.default_rng(1),
        task_shape={"needs_evidence": 0.9},
    )
    from app.orchestration.state import OrchestratorState

    restored = OrchestratorState.from_dict(shaped.to_dict())
    assert restored.needs_evidence == pytest.approx(0.9)
    assert np.allclose(restored.features(), shaped.features())


def test_task_shape_is_clamped():
    state = initial_state(
        task_complexity=0.5,
        budget_usd=1.0,
        latency_budget_ms=60_000.0,
        belief_dim=4,
        rng=np.random.default_rng(1),
        task_shape={"needs_evidence": 4.0, "needs_execution": -2.0},
    )
    assert state.needs_evidence == 1.0
    assert state.needs_execution == 0.0


# ------------------------------------------------------------------ profiles


def test_profile_round_trips_learned_parameters(session):
    svc = PolicyProfileService(session)
    trained = build()
    reward = run_to_completion(trained)
    svc.capture(trained, "router", episode_reward=reward)

    fresh = build()
    assert svc.apply_to(fresh, "router") is True
    np.testing.assert_allclose(fresh.policy.A, trained.policy.A)
    np.testing.assert_allclose(fresh.policy.b, trained.policy.b)


def test_missing_profile_reports_absence_rather_than_raising(session):
    assert PolicyProfileService(session).apply_to(build(), "never-trained") is False


def test_capture_accumulates_episode_statistics(session):
    svc = PolicyProfileService(session)
    for seed in (1, 2, 3):
        engine = build(seed=seed)
        svc.capture(engine, "router", episode_reward=run_to_completion(engine))

    profile = svc.get("router", "contextual_bandit")
    assert profile.episodes == 3
    assert profile.total_steps > 0
    assert profile.mean_episode_reward == pytest.approx(
        profile.cumulative_reward / 3
    )


def test_profiles_are_isolated_per_policy(session):
    svc = PolicyProfileService(session)
    bandit = build("contextual_bandit")
    svc.capture(bandit, "router", episode_reward=run_to_completion(bandit))

    # Same profile name, different policy: must not receive the bandit's weights.
    assert svc.apply_to(build("marl"), "router") is False


def test_stale_signature_is_refused_rather_than_silently_loaded(session):
    """LinUCB's ridge matrices are d x d, so a mismatched profile would corrupt the policy."""
    svc = PolicyProfileService(session)
    engine = build()
    svc.capture(engine, "router", episode_reward=run_to_completion(engine))

    profile = svc.get("router", "contextual_bandit")
    profile.signature = "contextual_bandit:d12:a8"
    session.commit()

    with pytest.raises(ProfileSignatureMismatch):
        svc.apply_to(build(), "router")


def test_signature_tracks_feature_dim():
    assert signature_for("marl") == f"marl:d{FEATURE_DIM}:a10"


def test_reset_discards_learned_parameters(session):
    svc = PolicyProfileService(session)
    engine = build()
    svc.capture(engine, "router", episode_reward=run_to_completion(engine))

    assert svc.reset("router", "contextual_bandit") is True
    assert svc.get("router", "contextual_bandit") is None
    assert svc.reset("router", "contextual_bandit") is False


def test_carried_profile_changes_scoring(session):
    """A loaded profile must actually influence decisions, not just deserialize cleanly.

    Asserted on the score vector rather than the trajectory: a tabular policy can carry real
    learning and still replay an identical path when the evaluation episode visits states it
    never saw during training.
    """
    svc = PolicyProfileService(session)
    for seed in range(6):
        trained = build(seed=seed)
        svc.capture(trained, "router", episode_reward=run_to_completion(trained))

    cold = build(seed=99)
    warm = build(seed=99)
    assert svc.apply_to(warm, "router") is True

    cold_scores = cold.policy.score_actions(cold.state)
    warm_scores = warm.policy.score_actions(warm.state)
    assert not np.allclose(cold_scores, warm_scores)


def _trajectory(engine: OrchestrationEngine):
    steps = []
    while not engine.done:
        steps.append(engine.step())
    return steps
