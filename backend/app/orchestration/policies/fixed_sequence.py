"""Fixed-sequence pipeline — the control that isolates *routing* from *decomposition*.

Comparing a learned router against a single undifferentiated agent conflates two claims:
that splitting work across specialist roles helps, and that *learning who to call next* helps.
A win could belong to either.

This policy holds the first constant. It walks the same six agents in a hardcoded order,
learns nothing, and adapts to nothing. Measured against a single agent it prices decomposition;
measured against the learned policies it prices routing, which is the claim this project makes.

The order is the pipeline a sensible engineer writes by hand: plan, gather, build, attack,
check, consolidate.
"""

from __future__ import annotations

import numpy as np

from ..actions import ACTIONS, ACTION_INDEX, Action, SINGLE_AGENT_ACTIONS
from ..state import OrchestratorState
from .base import Policy

DEFAULT_SEQUENCE: tuple[Action, ...] = (
    Action.INVOKE_PLANNER,
    Action.INVOKE_RESEARCHER,
    Action.INVOKE_EXECUTOR,
    Action.INVOKE_CRITIC,
    Action.INVOKE_VERIFIER,
    Action.INVOKE_MEMORY,
)


class FixedSequencePolicy(Policy):
    id = "fixed_sequence"
    label = "Fixed Pipeline"
    stage = 0
    family = "baseline"
    description = (
        "Walks planner, researcher, executor, critic, verifier and memory in a hardcoded cycle. "
        "No learning and no adaptation: the control that separates the value of decomposition "
        "from the value of learned routing."
    )

    def __init__(
        self,
        *,
        feature_dim: int,
        temperature: float = 0.05,
        sequence: tuple[Action, ...] = DEFAULT_SEQUENCE,
        **kwargs: object,
    ) -> None:
        super().__init__(feature_dim=feature_dim, temperature=temperature, **kwargs)
        self.sequence = tuple(sequence)
        self.cursor = 0

    def score_actions(self, state: OrchestratorState) -> np.ndarray:
        scores = np.full(len(ACTIONS), -6.0, dtype=float)

        # The "always orchestrate" bookend: escalate at the first opportunity, then run the
        # pipeline. Paired against single_agent, which never escalates, it brackets the gate.
        if not state.has_escalated:
            scores[ACTION_INDEX[Action.ESCALATE]] = 8.0
            scores[ACTION_INDEX[Action.INVOKE_GENERALIST]] = 2.0
            return scores

        # TERMINATE is only ever legal once the work is done or the resources are nearly gone,
        # so a pipeline with nothing left to do should take it.
        scores[ACTION_INDEX[Action.TERMINATE]] = 4.0

        # Never fans out: choosing a coalition size is a decision, and this policy makes none.
        scores[ACTION_INDEX[Action.RUN_PARALLEL]] = -8.0

        wanted = self.sequence[self.cursor % len(self.sequence)]
        scores[ACTION_INDEX[wanted]] = 8.0
        return scores

    def agent_preferences(self, state: OrchestratorState) -> dict[str, float]:
        wanted = self.sequence[self.cursor % len(self.sequence)]
        target = SINGLE_AGENT_ACTIONS.get(wanted)
        return {
            agent_id: (1.0 if agent_id == target else 0.05)
            for agent_id in SINGLE_AGENT_ACTIONS.values()
        }

    def update(
        self,
        state: OrchestratorState,
        action: Action,
        agents: list[str],
        reward: float,
        per_agent_reward: dict[str, float],
        next_state: OrchestratorState,
        done: bool,
    ) -> None:
        """Advances the pipeline. Deliberately ignores the reward — that is the whole point."""
        # The pre-escalation solo step is not part of the pipeline and must not consume a slot.
        if action in SINGLE_AGENT_ACTIONS and action is not Action.INVOKE_GENERALIST:
            self.cursor += 1

    def state_dict(self) -> dict:
        payload = super().state_dict()
        payload["cursor"] = self.cursor
        payload["sequence"] = [a.value for a in self.sequence]
        return payload

    def load_state_dict(self, payload: dict) -> None:
        super().load_state_dict(payload)
        self.cursor = int(payload.get("cursor", 0))
        stored = payload.get("sequence")
        if stored:
            self.sequence = tuple(Action(value) for value in stored)

    def diagnostics(self, state: OrchestratorState) -> dict:
        wanted = self.sequence[self.cursor % len(self.sequence)]
        return {
            "cursor": self.cursor,
            "next_in_pipeline": wanted.value,
            "cycle_length": len(self.sequence),
            "cycles_completed": self.cursor // len(self.sequence),
        }
