"""The orchestration engine.

One :class:`OrchestrationEngine` instance owns a single episode: its state, its policy, its RNG
stream and its accumulated totals. Everything is serializable, so an engine can be flushed to
SQLite after each step and rehydrated on demand without replaying history.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..config import Settings, get_settings
from .actions import ACTION_SPECS, ACTIONS, Action, SINGLE_AGENT_ACTIONS
from .agents import AGENTS
from .policies import DEFAULT_POLICY, Policy, create_policy
from .rewards import RewardModel
from .state import FEATURE_DIM, OrchestratorState, initial_state
from .transitions import TransitionModel, agents_for_action

ORCHESTRATOR_NODE = "orchestrator"


@dataclass
class RunConfig:
    task: str
    policy: str = DEFAULT_POLICY
    seed: int = 0
    task_complexity: float = 0.55
    budget_usd: float = 1.20
    latency_budget_ms: float = 90_000.0
    max_steps: int = 40
    belief_dim: int = 8
    stochasticity: float = 1.0
    confidence_target: float = 0.55
    """Reachable ceiling: agents emit correct evidence ~75% of the time, so p(truth) asymptotes
    near 0.65. Measured with tools/balance.py; 0.88 was unreachable in every episode."""
    verification_target: float = 0.75
    min_steps_before_terminate: int = 3
    """An orchestrator may not stop before it has produced anything."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RunConfig":
        known = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class StepResult:
    step: int
    action: str
    action_label: str
    agents: list[str]
    action_probability: float
    action_distribution: dict[str, float]
    transition_probability: float
    outcome: str
    reward: float
    cumulative_reward: float
    reward_breakdown: dict
    entropy_before: float
    entropy_after: float
    information_gain: float
    confidence: float
    cost_usd: float
    latency_ms: float
    tokens: int
    prev_state: dict
    state: dict
    reports: list[dict]
    messages: list[dict]
    diagnostics: dict
    done: bool
    termination_reason: str | None
    notes: str
    wall_clock_ms: float = 0.0
    legal_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class OrchestrationEngine:
    def __init__(self, config: RunConfig, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.policy: Policy = create_policy(config.policy, feature_dim=FEATURE_DIM)
        self.transition_model = TransitionModel(
            belief_dim=config.belief_dim, stochasticity=config.stochasticity
        )
        self.reward_model = RewardModel(self.settings.reward_weights)
        self.state = initial_state(
            task_complexity=config.task_complexity,
            budget_usd=config.budget_usd,
            latency_budget_ms=config.latency_budget_ms,
            belief_dim=config.belief_dim,
            rng=self.rng,
        )
        self.initial_state_payload = self.state.to_dict()
        self.cumulative_reward = 0.0
        self.total_cost = 0.0
        self.total_latency_ms = 0.0
        self.total_tokens = 0

    # ------------------------------------------------------------ properties
    @property
    def done(self) -> bool:
        return self.state.terminated

    # ---------------------------------------------------------------- step
    def legal_actions(self) -> list[Action]:
        legal = [a for a in ACTIONS if a not in {Action.RUN_PARALLEL, Action.TERMINATE}]
        if self.state.budget_remaining > 0.12 and self.state.latency_remaining > 0.1:
            legal.append(Action.RUN_PARALLEL)
        if self.state.step >= self.config.min_steps_before_terminate and self._may_terminate():
            legal.append(Action.TERMINATE)
        return legal

    def _may_terminate(self) -> bool:
        """Stopping is only a real choice once the work is done or the resources are nearly out.

        Without this the policy could quit at the first legal opportunity, which measurement
        showed it did in 97% of episodes.
        """
        state = self.state
        return (
            state.unresolved_subtasks == 0
            or state.budget_remaining < 0.25
            or state.latency_remaining < 0.25
        )

    def step(self) -> StepResult:
        if self.state.terminated:
            raise RuntimeError("Run has already terminated")

        started = time.perf_counter()
        prev_state = self.state.copy()
        legal = self.legal_actions()

        action, probability, distribution = self.policy.select(prev_state, legal, self.rng)

        coalition = None
        if action is Action.RUN_PARALLEL:
            coalition = self.policy.preferred_coalition(prev_state)
            if not coalition:
                coalition = agents_for_action(
                    action, prev_state, self.rng, self.policy.agent_preferences(prev_state)
                )

        outcome = self.transition_model.sample(
            prev_state,
            action,
            self.rng,
            preference=self.policy.agent_preferences(prev_state),
            agent_ids=coalition,
        )
        next_state = outcome.next_state
        agent_ids = [report.agent_id for report in outcome.reports]

        self._apply_termination_rules(next_state, action)

        breakdown = self.reward_model.compute(prev_state, outcome, action)
        if next_state.terminated and action is not Action.TERMINATE and breakdown.terminal == 0.0:
            # Environment-forced termination still earns (or loses) the terminal bonus.
            breakdown.terminal = self.settings.reward_weights.terminal * self.reward_model.terminal_value(
                next_state, action
            )
            breakdown.per_agent = self.reward_model.credit_assignment(breakdown, outcome)

        reward = breakdown.total
        self.cumulative_reward += reward
        self.total_cost += outcome.cost_usd
        self.total_latency_ms += outcome.latency_ms
        self.total_tokens += outcome.tokens

        for agent_id, share in breakdown.per_agent.items():
            history = next_state.agent_history.get(agent_id)
            if history is not None:
                history.cumulative_reward += share
                history.cumulative_information_gain += outcome.information_gain / max(
                    len(breakdown.per_agent), 1
                )

        self.policy.update(
            prev_state,
            action,
            agent_ids,
            reward,
            breakdown.per_agent,
            next_state,
            next_state.terminated,
        )

        self.state = next_state
        messages = self._build_messages(prev_state, action, agent_ids, outcome, probability)
        notes = self._notes(action, outcome, breakdown.total)

        return StepResult(
            step=next_state.step,
            action=action.value,
            action_label=ACTION_SPECS[action].label,
            agents=agent_ids,
            action_probability=probability,
            action_distribution=distribution,
            transition_probability=outcome.transition_probability,
            outcome=outcome.outcome,
            reward=reward,
            cumulative_reward=self.cumulative_reward,
            reward_breakdown=breakdown.to_dict(),
            entropy_before=outcome.entropy_before,
            entropy_after=outcome.entropy_after,
            information_gain=outcome.information_gain,
            confidence=next_state.confidence,
            cost_usd=outcome.cost_usd,
            latency_ms=outcome.latency_ms,
            tokens=outcome.tokens,
            prev_state=prev_state.to_dict(),
            state=next_state.to_dict(),
            reports=[r.to_dict() for r in outcome.reports],
            messages=messages,
            diagnostics=self.policy.diagnostics(prev_state),
            done=next_state.terminated,
            termination_reason=next_state.termination_reason,
            notes=notes,
            wall_clock_ms=(time.perf_counter() - started) * 1000.0,
            legal_actions=[a.value for a in legal],
        )

    # ---------------------------------------------------------- termination
    def _apply_termination_rules(self, state: OrchestratorState, action: Action) -> None:
        # Goal is evaluated first so a run that finished the job is labelled a win even when the
        # policy chose TERMINATE on the same step.
        goal = (
            state.confidence >= self.config.confidence_target
            and state.verification_score >= self.config.verification_target
            and state.unresolved_subtasks == 0
        )
        if goal:
            state.terminated = True
            state.termination_reason = "goal_reached"
            return
        if state.terminated:
            return
        if state.budget_remaining_usd <= 1e-6:
            state.terminated = True
            state.termination_reason = "budget_exhausted"
            return
        if state.latency_consumed_ms >= state.latency_budget_ms:
            state.terminated = True
            state.termination_reason = "latency_exhausted"
            return
        if state.step >= self.config.max_steps:
            state.terminated = True
            state.termination_reason = "step_limit"

    # -------------------------------------------------------------- messages
    def _build_messages(
        self,
        prev_state: OrchestratorState,
        action: Action,
        agent_ids: list[str],
        outcome,  # TransitionOutcome
        probability: float,
    ) -> list[dict]:
        step = self.state.step
        messages: list[dict] = []

        if action is Action.TERMINATE:
            messages.append(
                {
                    "step": step,
                    "sender": ORCHESTRATOR_NODE,
                    "receiver": ORCHESTRATOR_NODE,
                    "kind": "terminate",
                    "content": "Policy selected TERMINATE; emitting terminal reward.",
                    "weight": probability,
                }
            )
            return messages

        upstream = prev_state.last_agents or [ORCHESTRATOR_NODE]
        for receiver in agent_ids:
            for sender in upstream:
                if sender == receiver:
                    continue
                label = (
                    "Orchestrator dispatch"
                    if sender == ORCHESTRATOR_NODE
                    else f"{AGENTS[sender].label} handoff"
                )
                messages.append(
                    {
                        "step": step,
                        "sender": sender,
                        "receiver": receiver,
                        "kind": "handoff",
                        "content": (
                            f"{label} → {AGENTS[receiver].label}: "
                            f"u={prev_state.uncertainty:.2f}, "
                            f"open subtasks={prev_state.unresolved_subtasks}, "
                            f"p(a)={probability:.3f}"
                        ),
                        "weight": probability,
                    }
                )

        for report in outcome.reports:
            messages.append(
                {
                    "step": step,
                    "sender": report.agent_id,
                    "receiver": ORCHESTRATOR_NODE,
                    "kind": f"report:{report.outcome}",
                    "content": report.summary,
                    "weight": report.outcome_probability,
                }
            )

        if len(agent_ids) > 1:
            for i, a in enumerate(agent_ids):
                for b in agent_ids[i + 1 :]:
                    messages.append(
                        {
                            "step": step,
                            "sender": a,
                            "receiver": b,
                            "kind": "coordination",
                            "content": f"Parallel coalition sync between {AGENTS[a].label} and {AGENTS[b].label}.",
                            "weight": 0.5,
                        }
                    )
        return messages

    @staticmethod
    def _notes(action: Action, outcome, reward: float) -> str:
        if action is Action.TERMINATE:
            return "Terminal action selected by the policy."
        agents = ", ".join(AGENTS[r.agent_id].label for r in outcome.reports)
        return (
            f"{agents} → {outcome.outcome} "
            f"(ΔH={outcome.information_gain:+.3f} bits, R={reward:+.3f})"
        )

    # --------------------------------------------------------- serialization
    def snapshot(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
            "initial_state": self.initial_state_payload,
            "policy_state": self.policy.state_dict(),
            "rng_state": _serialize_rng(self.rng),
            "cumulative_reward": self.cumulative_reward,
            "total_cost": self.total_cost,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def restore(cls, snapshot: dict, settings: Settings | None = None) -> "OrchestrationEngine":
        config = RunConfig.from_dict(snapshot["config"])
        engine = cls(config, settings=settings)
        engine.state = OrchestratorState.from_dict(snapshot["state"])
        engine.initial_state_payload = snapshot.get("initial_state", engine.initial_state_payload)
        engine.policy.load_state_dict(snapshot.get("policy_state", {}))
        _deserialize_rng(engine.rng, snapshot.get("rng_state"))
        engine.cumulative_reward = float(snapshot.get("cumulative_reward", 0.0))
        engine.total_cost = float(snapshot.get("total_cost", 0.0))
        engine.total_latency_ms = float(snapshot.get("total_latency_ms", 0.0))
        engine.total_tokens = int(snapshot.get("total_tokens", 0))
        return engine

    # ------------------------------------------------------------ inspection
    def preview(self) -> dict:
        """Action distribution for the *current* state without advancing the episode."""
        legal = self.legal_actions()
        probabilities = self.policy.distribution(self.state, legal)
        return {
            "legal_actions": [a.value for a in legal],
            "distribution": {
                a.value: float(p) for a, p in zip(ACTIONS, probabilities, strict=True)
            },
            "diagnostics": self.policy.diagnostics(self.state),
            "expected_agents": {
                a.value: SINGLE_AGENT_ACTIONS.get(a) for a in ACTIONS
            },
            "preferred_coalition": self.policy.preferred_coalition(self.state),
        }


def _serialize_rng(rng: np.random.Generator) -> dict:
    state = rng.bit_generator.state
    return {
        "bit_generator": state["bit_generator"],
        "state": {k: str(v) for k, v in state["state"].items()},
        "has_uint32": int(state.get("has_uint32", 0)),
        "uinteger": int(state.get("uinteger", 0)),
    }


def _deserialize_rng(rng: np.random.Generator, payload: dict | None) -> None:
    if not payload:
        return
    rng.bit_generator.state = {
        "bit_generator": payload["bit_generator"],
        "state": {k: int(v) for k, v in payload["state"].items()},
        "has_uint32": int(payload.get("has_uint32", 0)),
        "uinteger": int(payload.get("uinteger", 0)),
    }
