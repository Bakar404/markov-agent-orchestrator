"""Stage 1 — contextual bandit orchestration (LinUCB).

Disjoint LinUCB (Li et al., 2010). One ridge-regression model per action over the shared state
feature vector φ(s):

    θ_a = A_a⁻¹ b_a
    UCB_a(s) = θ_aᵀφ(s) + α·sqrt(φ(s)ᵀ A_a⁻¹ φ(s))

The bandit treats each step as an independent decision — it optimizes immediate reward and has
no notion of a successor state. That limitation is exactly what stage 2 removes.
"""

from __future__ import annotations

import numpy as np

from ..actions import ACTIONS, ACTION_INDEX, Action, SINGLE_AGENT_ACTIONS
from ..state import OrchestratorState
from .base import Policy


class LinUCBPolicy(Policy):
    id = "contextual_bandit"
    label = "Contextual Bandit (LinUCB)"
    stage = 1
    family = "bandit"
    description = (
        "Disjoint LinUCB with one ridge model per action over the 12-dimensional state context. "
        "Optimizes immediate reward only; no credit flows backwards through time."
    )

    def __init__(
        self,
        *,
        feature_dim: int,
        temperature: float = 0.30,
        alpha: float = 0.85,
        ridge: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(feature_dim=feature_dim, temperature=temperature, **kwargs)
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        n = len(ACTIONS)
        self.A = np.stack([np.eye(feature_dim) * self.ridge for _ in range(n)])
        self.b = np.zeros((n, feature_dim), dtype=float)
        self.pulls = np.zeros(n, dtype=int)

    def _theta(self, index: int) -> np.ndarray:
        return np.linalg.solve(self.A[index], self.b[index])

    def score_actions(self, state: OrchestratorState) -> np.ndarray:
        x = state.features()
        scores = np.zeros(len(ACTIONS), dtype=float)
        for i in range(len(ACTIONS)):
            a_inv = np.linalg.inv(self.A[i])
            theta = a_inv @ self.b[i]
            mean = float(theta @ x)
            bonus = self.alpha * float(np.sqrt(max(x @ a_inv @ x, 0.0)))
            scores[i] = mean + bonus
        return scores

    def agent_preferences(self, state: OrchestratorState) -> dict[str, float]:
        scores = self.score_actions(state)
        shifted = scores - scores.min() + 1e-3
        return {
            agent_id: float(shifted[ACTION_INDEX[action]])
            for action, agent_id in SINGLE_AGENT_ACTIONS.items()
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
        x = state.features()
        i = ACTION_INDEX[action]
        self.A[i] += np.outer(x, x)
        self.b[i] += reward * x
        self.pulls[i] += 1

    def diagnostics(self, state: OrchestratorState) -> dict:
        x = state.features()
        rows = []
        for i, act in enumerate(ACTIONS):
            a_inv = np.linalg.inv(self.A[i])
            theta = a_inv @ self.b[i]
            rows.append(
                {
                    "action": act.value,
                    "expected_reward": float(theta @ x),
                    "exploration_bonus": self.alpha * float(np.sqrt(max(x @ a_inv @ x, 0.0))),
                    "pulls": int(self.pulls[i]),
                }
            )
        return {"model": "LinUCB", "alpha": self.alpha, "arms": rows}

    def state_dict(self) -> dict:
        payload = super().state_dict()
        payload.update(
            {
                "alpha": self.alpha,
                "ridge": self.ridge,
                "A": self.A.tolist(),
                "b": self.b.tolist(),
                "pulls": self.pulls.tolist(),
            }
        )
        return payload

    def load_state_dict(self, payload: dict) -> None:
        super().load_state_dict(payload)
        self.alpha = float(payload.get("alpha", self.alpha))
        self.ridge = float(payload.get("ridge", self.ridge))
        if "A" in payload:
            self.A = np.array(payload["A"], dtype=float)
        if "b" in payload:
            self.b = np.array(payload["b"], dtype=float)
        if "pulls" in payload:
            self.pulls = np.array(payload["pulls"], dtype=int)
