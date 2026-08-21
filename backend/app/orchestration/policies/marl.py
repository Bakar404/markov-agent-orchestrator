"""Stage 4 — multi-agent reinforcement learning orchestration.

Independent learners with VDN-style additive value factorization (Sunehag et al., 2017) and
difference-reward credit assignment in the spirit of COMA (Foerster et al., 2017):

    Q_tot(s, C) = Σ_{i∈C} Q_i(s)  +  Σ_{i∉C} B_i(s)

    δ      = r + γ·max_{C'} Q_tot(s', C') - Q_tot(s, C)
    w_i    ← w_i + η·δ·φ(s)                     (participating players, VDN gradient)
    v_i    ← v_i + 0.3·η·δ·φ(s)                 (abstaining players' baseline)
    a_i    ← a_i + η_a·(d_i - a_iᵀφ(s))·φ(s)    (difference-reward advantage head)

``B_i`` is the value of an agent *not* being invoked, which is what makes abstention a learned
decision rather than an omission, and ``a_i`` is the per-agent advantage used to break ties
between coalitions with the same additive value.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..actions import ACTIONS, ACTION_INDEX, AGENT_TO_ACTION, Action
from ..agents import AGENT_IDS
from ..state import OrchestratorState
from .base import Policy

MAX_COALITION_SIZE = 3


class MultiAgentRLPolicy(Policy):
    id = "marl"
    label = "Multi-Agent RL (VDN + difference rewards)"
    stage = 4
    family = "marl"
    description = (
        "Independent per-agent learners with additive value decomposition, learned abstention "
        "baselines, and a difference-reward advantage head for individual credit assignment."
    )

    def __init__(
        self,
        *,
        feature_dim: int,
        temperature: float = 0.24,
        learning_rate: float = 0.055,
        advantage_lr: float = 0.05,
        discount: float = 0.95,
        advantage_weight: float = 0.6,
        **kwargs: object,
    ) -> None:
        super().__init__(feature_dim=feature_dim, temperature=temperature, **kwargs)
        self.learning_rate = float(learning_rate)
        self.advantage_lr = float(advantage_lr)
        self.discount = float(discount)
        self.advantage_weight = float(advantage_weight)
        self.agent_ids = list(AGENT_IDS)
        n = len(self.agent_ids)
        self.q_weights = np.zeros((n, feature_dim), dtype=float)
        self.baseline_weights = np.zeros((n, feature_dim), dtype=float)
        self.advantage_weights = np.zeros((n, feature_dim), dtype=float)
        self.terminate_weights = np.zeros(feature_dim, dtype=float)
        self.episodes = 0
        self.updates = 0
        self.td_errors: list[float] = []

    # ------------------------------------------------------------- valuation
    def _heads(self, state: OrchestratorState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = state.features()
        return self.q_weights @ x, self.baseline_weights @ x, self.advantage_weights @ x

    def joint_value(self, state: OrchestratorState, indices: tuple[int, ...]) -> float:
        q, baseline, advantage = self._heads(state)
        members = list(indices)
        others = [i for i in range(len(self.agent_ids)) if i not in members]
        value = float(q[members].sum() + baseline[others].sum())
        value += self.advantage_weight * float(advantage[members].sum()) / max(len(members), 1)
        # Parallel fan-out is only worth it while there is budget to spend.
        value -= 0.3 * (len(members) - 1) * (1.0 - state.budget_remaining)
        return value

    def _all_coalitions(self) -> list[tuple[int, ...]]:
        n = len(self.agent_ids)
        coalitions: list[tuple[int, ...]] = []
        for size in range(1, min(MAX_COALITION_SIZE, n) + 1):
            coalitions.extend(combinations(range(n), size))
        return coalitions

    def _best_multi_coalition(self, state: OrchestratorState) -> tuple[tuple[int, ...], float]:
        best: tuple[int, ...] = (0, 1)
        best_value = -np.inf
        for coalition in self._all_coalitions():
            if len(coalition) < 2:
                continue
            value = self.joint_value(state, coalition)
            if value > best_value:
                best, best_value = coalition, value
        return best, float(best_value)

    # --------------------------------------------------------------- policy
    def score_actions(self, state: OrchestratorState) -> np.ndarray:
        scores = np.zeros(len(ACTIONS), dtype=float)
        for idx, agent_id in enumerate(self.agent_ids):
            scores[ACTION_INDEX[AGENT_TO_ACTION[agent_id]]] = self.joint_value(state, (idx,))
        _, parallel = self._best_multi_coalition(state)
        scores[ACTION_INDEX[Action.RUN_PARALLEL]] = parallel
        scores[ACTION_INDEX[Action.TERMINATE]] = float(
            self.terminate_weights @ state.features()
        ) + self._termination_readiness(state)
        return scores

    @staticmethod
    def _termination_readiness(state: OrchestratorState) -> float:
        ready = state.confidence * state.verification_score * (1.0 - state.unresolved_ratio)
        return float(-1.6 + 3.4 * ready + 1.8 * (1.0 - state.budget_remaining))

    def preferred_coalition(self, state: OrchestratorState) -> list[str]:
        indices, _ = self._best_multi_coalition(state)
        return [self.agent_ids[i] for i in indices]

    def agent_preferences(self, state: OrchestratorState) -> dict[str, float]:
        q, baseline, advantage = self._heads(state)
        edge = q - baseline + self.advantage_weight * advantage
        shifted = edge - edge.min() + 1e-3
        return {agent_id: float(shifted[i]) for i, agent_id in enumerate(self.agent_ids)}

    # ------------------------------------------------------------- learning
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
        norm = 1.0 + float(x @ x)
        self.updates += 1
        if done:
            self.episodes += 1

        if action is Action.TERMINATE or not agents:
            current = float(self.terminate_weights @ x) + self._termination_readiness(state)
            self.terminate_weights += self.learning_rate * (reward - current) * x / norm
            return

        members = [self.agent_ids.index(a) for a in agents if a in self.agent_ids]
        if not members:
            return

        bootstrap = 0.0
        if not done:
            bootstrap = max(self.joint_value(next_state, c) for c in self._all_coalitions())
        delta = reward + self.discount * bootstrap - self.joint_value(state, tuple(members))
        self.td_errors.append(float(delta))
        if len(self.td_errors) > 512:
            self.td_errors = self.td_errors[-512:]

        # VDN: the gradient of Q_tot wrt each participating Q_i is 1, so all share delta.
        for i in members:
            self.q_weights[i] += self.learning_rate * delta * x / norm
        for i in range(len(self.agent_ids)):
            if i not in members:
                self.baseline_weights[i] += 0.3 * self.learning_rate * delta * x / norm

        # Difference reward: agent i's marginal contribution to this step's reward.
        advantage_estimate = self.advantage_weights @ x
        for agent_id in agents:
            i = self.agent_ids.index(agent_id)
            d_i = per_agent_reward.get(agent_id, reward / len(agents)) - reward / len(agents)
            self.advantage_weights[i] += (
                self.advantage_lr * (d_i - advantage_estimate[i]) * x / norm
            )

    def diagnostics(self, state: OrchestratorState) -> dict:
        q, baseline, advantage = self._heads(state)
        coalition, value = self._best_multi_coalition(state)
        return {
            "model": "Independent learners + VDN mixing + difference rewards",
            "discount": self.discount,
            "updates": self.updates,
            "episodes": self.episodes,
            "mean_abs_td_error": float(np.mean(np.abs(self.td_errors))) if self.td_errors else 0.0,
            "players": [
                {
                    "agent": agent_id,
                    "q": float(q[i]),
                    "abstain_baseline": float(baseline[i]),
                    "advantage": float(advantage[i]),
                    "participation_edge": float(q[i] - baseline[i]),
                }
                for i, agent_id in enumerate(self.agent_ids)
            ],
            "best_coalition": {
                "agents": [self.agent_ids[i] for i in coalition],
                "value": value,
            },
        }

    def state_dict(self) -> dict:
        payload = super().state_dict()
        payload.update(
            {
                "learning_rate": self.learning_rate,
                "advantage_lr": self.advantage_lr,
                "discount": self.discount,
                "advantage_weight": self.advantage_weight,
                "q_weights": self.q_weights.tolist(),
                "baseline_weights": self.baseline_weights.tolist(),
                "advantage_weights": self.advantage_weights.tolist(),
                "terminate_weights": self.terminate_weights.tolist(),
                "episodes": self.episodes,
                "updates": self.updates,
                "td_errors": self.td_errors[-128:],
            }
        )
        return payload

    def load_state_dict(self, payload: dict) -> None:
        super().load_state_dict(payload)
        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        self.advantage_lr = float(payload.get("advantage_lr", self.advantage_lr))
        self.discount = float(payload.get("discount", self.discount))
        self.advantage_weight = float(payload.get("advantage_weight", self.advantage_weight))
        for key, attr in (
            ("q_weights", "q_weights"),
            ("baseline_weights", "baseline_weights"),
            ("advantage_weights", "advantage_weights"),
        ):
            if key in payload:
                setattr(self, attr, np.array(payload[key], dtype=float))
        if "terminate_weights" in payload:
            self.terminate_weights = np.array(payload["terminate_weights"], dtype=float)
        self.episodes = int(payload.get("episodes", 0))
        self.updates = int(payload.get("updates", 0))
        self.td_errors = [float(v) for v in payload.get("td_errors", [])]
