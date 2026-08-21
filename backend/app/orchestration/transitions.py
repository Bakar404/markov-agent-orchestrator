"""The transition kernel P(s' | s, a).

Nothing here is deterministic. For every invoked agent the kernel:

1. draws a competence sample ``p ~ Beta(alpha, beta)`` from that agent's success history,
   discounted by task complexity and residual uncertainty;
2. draws a uniform variate to realize one of ``success | partial | failure`` and records the
   probability of the realized branch (this is the ``transition_probability`` in the trace);
3. draws cost, latency and token consumption from log-normal distributions;
4. folds sampled Dirichlet evidence into the belief — correct evidence concentrates on the
   latent hypothesis, noisy evidence spreads over the alternatives.

Two invocations of the same action from the same state therefore land in different successor
states, which is the property the whole platform is built to visualize.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actions import Action, SINGLE_AGENT_ACTIONS
from .agents import AGENTS, AgentSpec
from .entropy import belief_entropy
from .state import OrchestratorState

OUTCOMES = ("success", "partial", "failure")


@dataclass
class AgentReport:
    """What a single agent invocation produced within a step."""

    agent_id: str
    outcome: str
    outcome_probability: float
    competence_sample: float
    cost_usd: float
    latency_ms: float
    tokens: int
    evidence_mass: float
    correct_evidence: bool
    summary: str

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "outcome": self.outcome,
            "outcome_probability": self.outcome_probability,
            "competence_sample": self.competence_sample,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "evidence_mass": self.evidence_mass,
            "correct_evidence": self.correct_evidence,
            "summary": self.summary,
        }


@dataclass
class TransitionOutcome:
    next_state: OrchestratorState
    reports: list[AgentReport]
    outcome: str
    transition_probability: float
    entropy_before: float
    entropy_after: float
    information_gain: float
    cost_usd: float
    latency_ms: float
    tokens: int
    duplicate_penalty: float
    resolved_subtasks: int
    deltas: dict[str, float] = field(default_factory=dict)


def _lognormal(rng: np.random.Generator, mean: float, sigma: float) -> float:
    """Positive draw with median ``mean``."""
    return float(mean * np.exp(rng.normal(0.0, sigma) - 0.0))


def agents_for_action(
    action: Action,
    state: OrchestratorState,
    rng: np.random.Generator,
    preference: dict[str, float] | None = None,
) -> list[str]:
    """Resolve an action to the concrete agent(s) that will be invoked."""
    if action in SINGLE_AGENT_ACTIONS:
        return [SINGLE_AGENT_ACTIONS[action]]
    if action is Action.TERMINATE:
        return []

    # RUN_PARALLEL: build a coalition. Preference scores come from the policy when it exposes
    # them (Markov game / MARL); otherwise fall back to a need-based heuristic.
    scores = dict(preference or {})
    if not scores:
        scores = {
            "planner": 0.4 + 0.6 * state.unresolved_ratio,
            "researcher": 0.3 + 0.9 * state.uncertainty,
            "critic": 0.2 + 0.7 * (1.0 - state.quality),
            "verifier": 0.2 + 0.8 * (1.0 - state.verification_score),
            "memory": 0.2 + 0.7 * (1.0 - state.memory_coverage),
            "executor": 0.25 + 0.7 * state.unresolved_ratio,
        }
    keys = list(scores.keys())
    weights = np.array([max(scores[k], 1e-6) for k in keys], dtype=float)
    weights = weights / weights.sum()
    size = 2 if rng.random() < 0.7 else 3
    size = min(size, len(keys))
    chosen = rng.choice(len(keys), size=size, replace=False, p=weights)
    return [keys[int(i)] for i in chosen]


class TransitionModel:
    """Samples the successor state. Holds no mutable state of its own."""

    def __init__(self, *, belief_dim: int, stochasticity: float = 1.0) -> None:
        self.belief_dim = belief_dim
        self.stochasticity = float(np.clip(stochasticity, 0.05, 3.0))

    # ------------------------------------------------------------------ core
    def sample(
        self,
        state: OrchestratorState,
        action: Action,
        rng: np.random.Generator,
        *,
        preference: dict[str, float] | None = None,
        agent_ids: list[str] | None = None,
    ) -> TransitionOutcome:
        entropy_before = state.entropy
        next_state = state.copy()
        next_state.step = state.step + 1

        if action is Action.TERMINATE:
            next_state.terminated = True
            next_state.termination_reason = "policy_terminate"
            return TransitionOutcome(
                next_state=next_state,
                reports=[],
                outcome="terminal",
                transition_probability=1.0,
                entropy_before=entropy_before,
                entropy_after=entropy_before,
                information_gain=0.0,
                cost_usd=0.0,
                latency_ms=0.0,
                tokens=0,
                duplicate_penalty=0.0,
                resolved_subtasks=0,
                deltas={},
            )

        resolved_agents = agent_ids if agent_ids else agents_for_action(action, state, rng, preference)
        reports: list[AgentReport] = []
        belief = next_state.belief_array.copy()

        total_cost = 0.0
        max_latency = 0.0
        sum_latency = 0.0
        total_tokens = 0
        quality_delta = 0.0
        verification_delta = 0.0
        memory_delta = 0.0
        resolved = 0

        for agent_id in resolved_agents:
            spec = AGENTS[agent_id]
            report = self._invoke(spec, next_state, rng)
            reports.append(report)

            total_cost += report.cost_usd
            sum_latency += report.latency_ms
            max_latency = max(max_latency, report.latency_ms)
            total_tokens += report.tokens

            belief = self._apply_evidence(belief, spec, report, next_state, rng)

            scale = {"success": 1.0, "partial": 0.45, "failure": -0.25}[report.outcome]
            quality_delta += spec.quality_gain * scale * float(rng.uniform(0.7, 1.3))
            verification_delta += spec.verification_gain * max(scale, 0.0) * float(
                rng.uniform(0.7, 1.3)
            )
            memory_delta += spec.memory_gain * max(scale, 0.05) * float(rng.uniform(0.8, 1.2))

            if spec.subtask_resolution > 0 and report.outcome != "failure":
                expected = spec.subtask_resolution * (1.0 if report.outcome == "success" else 0.5)
                resolved += int(rng.binomial(max(next_state.unresolved_subtasks, 0), min(expected, 0.95)))

            history = next_state.agent_history[agent_id]
            history.invocations += 1
            history.last_step = next_state.step
            history.cumulative_cost += report.cost_usd
            history.cumulative_tokens += report.tokens
            if report.outcome == "success":
                history.alpha += 1.0
                history.successes += 1
            elif report.outcome == "partial":
                history.alpha += 0.5
                history.beta += 0.5
                history.partials += 1
            else:
                history.beta += 1.0
                history.failures += 1

        # Latency for a coalition is the slowest member plus a coordination overhead.
        latency_ms = max_latency + 0.12 * (sum_latency - max_latency)

        next_state.belief = [float(v) for v in belief]
        next_state.budget_spent_usd = state.budget_spent_usd + total_cost
        next_state.latency_consumed_ms = state.latency_consumed_ms + latency_ms
        next_state.tokens_consumed = state.tokens_consumed + total_tokens
        next_state.quality = float(np.clip(state.quality + quality_delta, 0.0, 1.0))
        next_state.verification_score = float(
            np.clip(state.verification_score + verification_delta, 0.0, 1.0)
        )
        next_state.memory_coverage = float(np.clip(state.memory_coverage + memory_delta, 0.0, 1.0))

        if "planner" in resolved_agents:
            next_state.total_subtasks, next_state.unresolved_subtasks = self._replan(
                next_state, reports, rng
            )

        resolved = min(resolved, next_state.unresolved_subtasks)
        next_state.unresolved_subtasks = max(next_state.unresolved_subtasks - resolved, 0)

        entropy_after = belief_entropy(np.asarray(next_state.belief, dtype=float))
        information_gain = entropy_before - entropy_after

        duplicate_penalty, signatures = self._duplicate_pressure(
            state, resolved_agents, information_gain
        )
        next_state.invocation_signatures = signatures
        next_state.duplicate_pressure = float(
            np.clip(0.72 * state.duplicate_pressure + duplicate_penalty, 0.0, 1.0)
        )
        next_state.last_agents = list(resolved_agents)

        outcome = self._aggregate_outcome(reports)
        transition_probability = float(np.prod([r.outcome_probability for r in reports])) if reports else 1.0

        return TransitionOutcome(
            next_state=next_state,
            reports=reports,
            outcome=outcome,
            transition_probability=transition_probability,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            information_gain=information_gain,
            cost_usd=total_cost,
            latency_ms=latency_ms,
            tokens=total_tokens,
            duplicate_penalty=duplicate_penalty,
            resolved_subtasks=resolved,
            deltas={
                "quality": next_state.quality - state.quality,
                "verification_score": next_state.verification_score - state.verification_score,
                "memory_coverage": next_state.memory_coverage - state.memory_coverage,
                "confidence": next_state.confidence - state.confidence,
                "unresolved_subtasks": float(next_state.unresolved_subtasks - state.unresolved_subtasks),
            },
        )

    # ------------------------------------------------------------- internals
    def _invoke(
        self, spec: AgentSpec, state: OrchestratorState, rng: np.random.Generator
    ) -> AgentReport:
        history = state.agent_history[spec.id]
        competence = float(rng.beta(max(history.alpha, 1e-3), max(history.beta, 1e-3)))

        # Harder tasks and higher residual uncertainty depress the realized success chance.
        difficulty = 0.55 * state.task_complexity + 0.35 * state.uncertainty
        p_success = float(np.clip(competence * (1.0 - 0.55 * difficulty), 0.02, 0.97))
        p_partial = float(np.clip((1.0 - p_success) * rng.uniform(0.45, 0.75), 0.01, 0.9))
        p_failure = float(max(1.0 - p_success - p_partial, 1e-4))

        u = float(rng.random())
        if u < p_success:
            outcome, probability = "success", p_success
        elif u < p_success + p_partial:
            outcome, probability = "partial", p_partial
        else:
            outcome, probability = "failure", p_failure

        sigma = 0.28 * self.stochasticity
        cost = _lognormal(rng, spec.base_cost_usd, sigma) * (1.0 + 0.4 * state.task_complexity)
        latency = _lognormal(rng, spec.base_latency_ms, sigma) * (1.0 + 0.3 * state.task_complexity)
        tokens = int(max(_lognormal(rng, spec.base_tokens, sigma * 0.8), 1))

        # A failed call still burns budget but produces little usable evidence.
        evidence_scale = {"success": 1.0, "partial": 0.45, "failure": 0.1}[outcome]
        evidence_mass = spec.evidence_strength * evidence_scale * float(rng.uniform(0.75, 1.35))
        correct = outcome != "failure" and rng.random() < (0.55 + 0.4 * competence)

        summary = self._summary(spec, outcome, state)
        return AgentReport(
            agent_id=spec.id,
            outcome=outcome,
            outcome_probability=probability,
            competence_sample=competence,
            cost_usd=cost,
            latency_ms=latency,
            tokens=tokens,
            evidence_mass=evidence_mass,
            correct_evidence=bool(correct),
            summary=summary,
        )

    def _apply_evidence(
        self,
        belief: np.ndarray,
        spec: AgentSpec,
        report: AgentReport,
        state: OrchestratorState,
        rng: np.random.Generator,
    ) -> np.ndarray:
        dim = belief.size
        truth = state.latent_hypothesis
        update = np.zeros(dim, dtype=float)

        if report.correct_evidence:
            update[truth] += report.evidence_mass
        else:
            wrong = [i for i in range(dim) if i != truth]
            update[int(rng.choice(wrong))] += report.evidence_mass * 0.7

        # Every agent also emits diffuse noise, but its weight decays as evidence accumulates:
        # one late hallucination should not re-flatten a well-supported posterior, while early
        # exploration stays genuinely uncertain.
        observations = max(float(belief.sum()) - dim, 0.0)
        noise_decay = 4.0 / (4.0 + observations)
        noise = (
            rng.dirichlet(np.full(dim, 0.9))
            * spec.noise_strength
            * self.stochasticity
            * noise_decay
        )
        update += noise

        return belief + update

    def _replan(
        self,
        state: OrchestratorState,
        reports: list[AgentReport],
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        planner_report = next((r for r in reports if r.agent_id == "planner"), None)
        if planner_report is None:
            return state.total_subtasks, state.unresolved_subtasks

        if planner_report.outcome == "success":
            # A good plan collapses ambiguous work into fewer concrete subtasks.
            delta = -int(rng.integers(0, 3))
        elif planner_report.outcome == "partial":
            delta = int(rng.integers(-1, 2))
        else:
            # A bad plan discovers work that was not previously modeled.
            delta = int(rng.integers(1, 4))

        unresolved = int(np.clip(state.unresolved_subtasks + delta, 0, 24))
        total = max(state.total_subtasks, unresolved)
        return total, unresolved

    def _duplicate_pressure(
        self,
        state: OrchestratorState,
        agent_ids: list[str],
        information_gain: float,
    ) -> tuple[float, list[str]]:
        """Penalize re-running an agent that produced no new information."""
        signature = "+".join(sorted(agent_ids))
        recent = list(state.invocation_signatures)[-4:]
        repeats = recent.count(signature)
        stale = max(0.0, 0.08 - information_gain) / 0.08
        overlap = len(set(agent_ids) & set(state.last_agents)) / max(len(agent_ids), 1)
        penalty = float(np.clip(0.35 * repeats + 0.45 * stale * overlap, 0.0, 1.0))
        signatures = (state.invocation_signatures + [signature])[-12:]
        return penalty, signatures

    @staticmethod
    def _aggregate_outcome(reports: list[AgentReport]) -> str:
        if not reports:
            return "noop"
        outcomes = [r.outcome for r in reports]
        if all(o == "success" for o in outcomes):
            return "success"
        if all(o == "failure" for o in outcomes):
            return "failure"
        return "partial"

    @staticmethod
    def _summary(spec: AgentSpec, outcome: str, state: OrchestratorState) -> str:
        verb = {
            "planner": ("decomposed the frontier into concrete subtasks", "produced a partial plan", "returned an incoherent plan"),
            "researcher": ("retrieved corroborating evidence", "retrieved weakly relevant sources", "retrieved nothing usable"),
            "critic": ("identified and closed a failure mode", "raised an unresolved objection", "misread the artifact"),
            "verifier": ("verified the artifact against constraints", "verified part of the artifact", "verification failed"),
            "memory": ("recalled relevant prior context", "recalled loosely related context", "found no prior context"),
            "executor": ("produced the artifact increment", "produced a partial increment", "the execution attempt failed"),
        }[spec.id]
        index = {"success": 0, "partial": 1, "failure": 2}[outcome]
        return f"{spec.label} {verb[index]} (step {state.step}, u={state.uncertainty:.2f})."
