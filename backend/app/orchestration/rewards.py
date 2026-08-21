"""The reward function.

    R(s, a, s') = w_q·Δquality
                + w_v·Δverification
                + w_i·InformationGain
                + w_p·ProgressRatio
                - w_c·NormalizedCost
                - w_l·NormalizedLatency
                - w_d·DuplicateWork
                + TerminalBonus

``InformationGain = H(belief_before) - H(belief_after)`` in bits, taken directly from the
transition outcome so the UI can show the same arithmetic it displays in the metrics panel.
Every term is returned separately: the reward dashboard renders the decomposition, and the
Markov-game / MARL policies use the per-term split for credit assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import RewardWeights
from .actions import Action
from .state import OrchestratorState
from .transitions import TransitionOutcome


@dataclass
class RewardBreakdown:
    quality: float = 0.0
    verification: float = 0.0
    information_gain: float = 0.0
    progress: float = 0.0
    cost: float = 0.0
    latency: float = 0.0
    duplicate: float = 0.0
    terminal: float = 0.0
    per_agent: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (
            self.quality
            + self.verification
            + self.information_gain
            + self.progress
            + self.cost
            + self.latency
            + self.duplicate
            + self.terminal
        )

    def to_dict(self) -> dict:
        return {
            "quality": self.quality,
            "verification": self.verification,
            "information_gain": self.information_gain,
            "progress": self.progress,
            "cost": self.cost,
            "latency": self.latency,
            "duplicate": self.duplicate,
            "terminal": self.terminal,
            "total": self.total,
            "per_agent": dict(self.per_agent),
        }


class RewardModel:
    def __init__(self, weights: RewardWeights) -> None:
        self.w = weights

    def compute(
        self,
        prev_state: OrchestratorState,
        outcome: TransitionOutcome,
        action: Action,
    ) -> RewardBreakdown:
        next_state = outcome.next_state
        breakdown = RewardBreakdown()

        breakdown.quality = self.w.quality * (next_state.quality - prev_state.quality)
        breakdown.verification = self.w.verification * (
            next_state.verification_score - prev_state.verification_score
        )
        breakdown.information_gain = self.w.information_gain * outcome.information_gain

        progress = outcome.resolved_subtasks / max(prev_state.total_subtasks, 1)
        breakdown.progress = self.w.progress * progress

        normalized_cost = outcome.cost_usd / max(prev_state.budget_total_usd, 1e-9)
        breakdown.cost = -self.w.cost * normalized_cost

        normalized_latency = outcome.latency_ms / max(prev_state.latency_budget_ms, 1e-9)
        breakdown.latency = -self.w.latency * normalized_latency

        breakdown.duplicate = -self.w.duplicate * outcome.duplicate_penalty

        if next_state.terminated:
            breakdown.terminal = self.w.terminal * self.terminal_value(next_state, action)

        breakdown.per_agent = self.credit_assignment(breakdown, outcome)
        return breakdown

    def terminal_value(self, state: OrchestratorState, action: Action) -> float:
        """Terminal bonus rewards *finishing well*, and punishes bailing out early."""
        readiness = (
            0.40 * state.confidence
            + 0.30 * state.quality
            + 0.30 * state.verification_score
        )
        completion = 1.0 - state.unresolved_ratio
        value = readiness * (0.35 + 0.65 * completion)

        if action is Action.TERMINATE and state.unresolved_ratio > 0.5:
            value -= 0.45 * state.unresolved_ratio
        if state.termination_reason == "budget_exhausted":
            value -= 0.35
        if state.termination_reason == "latency_exhausted":
            value -= 0.25
        if state.termination_reason == "goal_reached":
            # Without this, finishing fast and cheap dominates finishing well.
            value += 0.5
        return float(np.clip(value, -1.0, 1.0))

    @staticmethod
    def credit_assignment(breakdown: RewardBreakdown, outcome: TransitionOutcome) -> dict[str, float]:
        """Split the step reward across the invoked coalition.

        Shares are proportional to each agent's realized evidence contribution net of its own
        cost, which is the difference-reward signal the MARL policy learns from.
        """
        if not outcome.reports:
            return {}
        contributions = np.array(
            [max(r.evidence_mass, 0.0) + 0.5 * (r.outcome == "success") for r in outcome.reports],
            dtype=float,
        )
        if contributions.sum() <= 1e-9:
            contributions = np.ones_like(contributions)
        shares = contributions / contributions.sum()
        total = breakdown.total
        return {
            report.agent_id: float(total * share)
            for report, share in zip(outcome.reports, shares, strict=True)
        }
