"""Single-agent control — the arm that uses no orchestration at all.

Every other policy answers "who should act next". This one refuses the question: it invokes one
agent, repeatedly, and never fans out. It is what you get *without* the framework, expressed as
a policy so a control run flows through the same transition kernel, cost accounting, trace
persistence and UI as the arms it is being compared against.

Note what this cannot tell you. The reward function rewards belief collapse and subtask
resolution, both of which presume decomposition, so a control will score poorly on internal
reward whether or not its answer was any good. Compare arms on cost, latency, tokens and an
external quality judgment — not on ``cumulative_reward``.
"""

from __future__ import annotations

import numpy as np

from ..actions import ACTIONS, ACTION_INDEX, Action, SINGLE_AGENT_ACTIONS
from ..state import OrchestratorState
from .base import Policy

AGENT_TO_ACTION: dict[str, Action] = {
    agent_id: action for action, agent_id in SINGLE_AGENT_ACTIONS.items()
}


class SingleAgentPolicy(Policy):
    id = "single_agent"
    label = "Single Agent (control)"
    stage = 0
    family = "baseline"
    description = (
        "Invokes one agent every step and never fans out. The no-orchestration control: "
        "compare arms on cost, latency and external quality, not on internal reward."
    )

    def __init__(
        self,
        *,
        feature_dim: int,
        temperature: float = 0.05,
        agent_id: str = "generalist",
        **kwargs: object,
    ) -> None:
        super().__init__(feature_dim=feature_dim, temperature=temperature, **kwargs)
        if agent_id not in AGENT_TO_ACTION:
            known = ", ".join(sorted(AGENT_TO_ACTION))
            raise ValueError(f"Unknown agent '{agent_id}'. Available: {known}")
        self.agent_id = agent_id

    def score_actions(self, state: OrchestratorState) -> np.ndarray:
        scores = np.full(len(ACTIONS), -8.0, dtype=float)
        scores[ACTION_INDEX[AGENT_TO_ACTION[self.agent_id]]] = 8.0
        # The "never orchestrate" bookend. Pair it with fixed_sequence, which always escalates,
        # to bracket whatever the learned gate decides.
        scores[ACTION_INDEX[Action.ESCALATE]] = -12.0
        # Stops as soon as the environment permits; a control does not budget-hunt.
        scores[ACTION_INDEX[Action.TERMINATE]] = 6.0
        return scores

    def agent_preferences(self, state: OrchestratorState) -> dict[str, float]:
        return {
            agent_id: (1.0 if agent_id == self.agent_id else 0.0)
            for agent_id in SINGLE_AGENT_ACTIONS.values()
        }

    def preferred_coalition(self, state: OrchestratorState) -> list[str]:
        return [self.agent_id]

    def state_dict(self) -> dict:
        payload = super().state_dict()
        payload["agent_id"] = self.agent_id
        return payload

    def load_state_dict(self, payload: dict) -> None:
        super().load_state_dict(payload)
        self.agent_id = str(payload.get("agent_id", self.agent_id))

    def diagnostics(self, state: OrchestratorState) -> dict:
        return {"agent_id": self.agent_id, "routing": "none"}
