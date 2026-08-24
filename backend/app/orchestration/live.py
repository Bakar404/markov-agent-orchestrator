"""Live mode: real agent invocations instead of sampled ones.

Sim mode draws an outcome from ``Beta(alpha, beta)`` and calls it a day. Live mode splits the
step in two so a real model can sit in the middle:

1. ``open`` — the policy picks the action and the agent(s). We hand back a *brief* describing
   what that agent is supposed to do. No state advances.
2. ``report`` — the caller returns what the agent actually produced. We convert it into
   :class:`AgentReport` objects and push them through the same transition kernel sim mode uses,
   so reward, entropy, credit assignment and persistence are untouched.

The grading problem is the interesting part. Sim mode knows ``latent_hypothesis`` and can score
evidence as correct or not. Real tasks have no hidden label, so live reports carry a
``claimed_hypothesis`` instead: belief mass follows what each agent argued for, and truth has to
emerge from agreement between independent agents rather than from an oracle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .actions import Action
from .agents import AGENTS, AgentSpec
from .state import OrchestratorState
from .transitions import OUTCOMES, AgentReport

OUTCOME_SCALE = {"success": 1.0, "partial": 0.45, "failure": 0.1}

ROLE_INSTRUCTIONS: dict[str, str] = {
    "planner": (
        "Decompose the task into concrete subtasks and name the competing hypotheses that could "
        "each be the answer. Do not attempt the work itself."
    ),
    "researcher": (
        "Gather evidence bearing on the open hypotheses. Cite what you actually consulted; say "
        "plainly when you could not find support rather than inferring it."
    ),
    "critic": (
        "Attack the leading hypothesis. Look for the failure mode nobody has considered. "
        "Raising uncertainty is a valid result here."
    ),
    "verifier": (
        "Independently check the leading hypothesis against constraints and sources. Do not "
        "defer to earlier agents — disagreement is signal, not error."
    ),
    "memory": (
        "Consolidate what is established so far, flag contradictions between prior steps, and "
        "surface anything already answered that is being re-derived."
    ),
    "executor": (
        "Carry out the highest-value unresolved subtask and report the concrete result."
    ),
}


@dataclass
class AgentBrief:
    """What the orchestrator hands to a real agent for one invocation."""

    agent_id: str
    label: str
    role: str
    instruction: str
    hypotheses: list[str]
    context: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PendingStep:
    """A step that has been decided by the policy but not yet realized."""

    token: str
    run_id: str
    step: int
    action: str
    agent_ids: list[str]
    action_probability: float
    action_distribution: dict[str, float]
    briefs: list[AgentBrief] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "run_id": self.run_id,
            "step": self.step,
            "action": self.action,
            "agents": self.agent_ids,
            "action_probability": self.action_probability,
            "action_distribution": self.action_distribution,
            "briefs": [brief.to_dict() for brief in self.briefs],
        }


def default_hypotheses(belief_dim: int) -> list[str]:
    return [f"hypothesis-{i}" for i in range(belief_dim)]


def build_brief(
    spec: AgentSpec,
    state: OrchestratorState,
    task: str,
    hypotheses: list[str],
) -> AgentBrief:
    """Describe one agent's job for this step, grounded in the current state."""
    belief = state.belief_probabilities
    ranked = sorted(
        ({"index": i, "label": hypotheses[i], "probability": float(belief[i])} for i in range(len(hypotheses))),
        key=lambda h: h["probability"],
        reverse=True,
    )
    return AgentBrief(
        agent_id=spec.id,
        label=spec.label,
        role=spec.role,
        instruction=ROLE_INSTRUCTIONS.get(spec.id, spec.description),
        hypotheses=list(hypotheses),
        context={
            "task": task,
            "step": state.step,
            "belief_ranked": ranked,
            "leading_hypothesis": ranked[0] if ranked else None,
            "entropy_bits": round(state.entropy, 4),
            "confidence": round(state.confidence, 4),
            "quality": round(state.quality, 4),
            "verification_score": round(state.verification_score, 4),
            "unresolved_subtasks": state.unresolved_subtasks,
            "budget_remaining_usd": round(state.budget_remaining_usd, 4),
            "prior_success_rate": round(state.agent_history[spec.id].success_rate, 4),
        },
    )


def report_from_response(
    spec: AgentSpec,
    state: OrchestratorState,
    payload: dict,
    *,
    belief_dim: int,
) -> AgentReport:
    """Convert one real agent response into the report the transition kernel consumes."""
    outcome = str(payload.get("outcome", "partial")).lower()
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got '{outcome}'")

    confidence = float(np.clip(float(payload.get("confidence", 0.5)), 0.0, 1.0))

    claimed = payload.get("claimed_hypothesis")
    if claimed is None:
        # An agent that argued for nothing still costs money and still adds noise; it simply
        # concentrates no mass. Point it at the current leader with zero-ish weight instead.
        claimed_index = int(np.argmax(state.belief_probabilities))
        confidence = min(confidence, 0.15)
    else:
        claimed_index = int(claimed)
        if not 0 <= claimed_index < belief_dim:
            raise ValueError(f"claimed_hypothesis {claimed_index} outside [0, {belief_dim})")

    history = state.agent_history[spec.id]
    # No branch was sampled, so the honest stand-in for P(realized outcome) is the agent's prior
    # success rate — how likely this result was before we saw it.
    prior = float(history.success_rate)
    outcome_probability = {
        "success": prior,
        "partial": max(1.0 - prior, 1e-4) * 0.6,
        "failure": max(1.0 - prior, 1e-4) * 0.4,
    }[outcome]

    tokens = int(max(payload.get("tokens") or spec.base_tokens, 1))
    latency_ms = float(max(payload.get("latency_ms") or spec.base_latency_ms, 0.0))
    cost_usd = payload.get("cost_usd")
    if cost_usd is None:
        # Derive from measured tokens at the agent's configured rate.
        cost_usd = spec.base_cost_usd * (tokens / max(spec.base_tokens, 1))
    cost_usd = float(max(cost_usd, 0.0))

    evidence_mass = spec.evidence_strength * OUTCOME_SCALE[outcome] * (0.4 + 0.6 * confidence)

    response = str(payload.get("response", "") or "")
    summary = str(payload.get("summary", "") or "").strip()
    if not summary:
        summary = response.strip().splitlines()[0][:160] if response.strip() else f"{spec.label}: {outcome}"

    return AgentReport(
        agent_id=spec.id,
        outcome=outcome,
        outcome_probability=float(np.clip(outcome_probability, 1e-4, 1.0)),
        competence_sample=prior,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        tokens=tokens,
        evidence_mass=float(evidence_mass),
        correct_evidence=outcome != "failure" and confidence >= 0.5,
        summary=summary,
        source="live",
        claimed_hypothesis=claimed_index,
        response_excerpt=response[:2000],
    )


def coalition_for(action: Action, agent_ids: list[str]) -> list[str]:
    unknown = [a for a in agent_ids if a not in AGENTS]
    if unknown:
        raise ValueError(f"unknown agent id(s): {', '.join(unknown)}")
    return agent_ids
