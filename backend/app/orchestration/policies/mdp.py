"""Stage 2 — Markov Decision Process orchestration.

Tabular Q-learning over a discretized state, with a linear function approximator blended in so
the policy still produces sensible scores for states it has never visited:

    Q(s,a) = Q_table[bucket(s), a] + w_aᵀφ(s)

    Q_table ← Q_table + η·(r + γ·max_a' Q(s',a') - Q(s,a))
    w_a     ← w_a + η_w·δ·φ(s)

Unlike the bandit, the TD target contains ``max_a' Q(s',a')``, so value propagates backwards
across steps and the policy can pay a cost now for a better terminal bonus later.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..actions import ACTIONS, ACTION_INDEX, Action, SINGLE_AGENT_ACTIONS
from ..state import OrchestratorState
from .base import Policy


class MDPQLearningPolicy(Policy):
    id = "mdp"
    label = "MDP (Q-Learning)"
    stage = 2
    family = "mdp"
    description = (
        "Tabular Q-learning over discretized states blended with a per-action linear model, "
        "Boltzmann exploration, and a bootstrapped TD target that propagates terminal value."
    )

    def __init__(
        self,
        *,
        feature_dim: int,
        temperature: float = 0.28,
        learning_rate: float = 0.35,
        approximator_lr: float = 0.05,
        discount: float = 0.94,
        bins: int = 4,
        optimistic_init: float = 0.25,
        **kwargs: object,
    ) -> None:
        super().__init__(feature_dim=feature_dim, temperature=temperature, **kwargs)
        self.learning_rate = float(learning_rate)
        self.approximator_lr = float(approximator_lr)
        self.discount = float(discount)
        self.bins = int(bins)
        self.optimistic_init = float(optimistic_init)
        n = len(ACTIONS)
        self.q: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.full(n, self.optimistic_init, dtype=float)
        )
        self.weights = np.zeros((n, feature_dim), dtype=float)
        self.visits: dict[tuple[int, ...], int] = defaultdict(int)
        self.td_errors: list[float] = []

    def _q_row(self, state: OrchestratorState) -> np.ndarray:
        return self.q[state.discretize(self.bins)]

    def score_actions(self, state: OrchestratorState) -> np.ndarray:
        return self._q_row(state) + self.weights @ state.features()

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
        key = state.discretize(self.bins)
        i = ACTION_INDEX[action]
        x = state.features()

        current = self.q[key][i] + float(self.weights[i] @ x)
        bootstrap = 0.0 if done else float(np.max(self.score_actions(next_state)))
        target = reward + self.discount * bootstrap
        delta = target - current

        self.visits[key] += 1
        decay = self.learning_rate / (1.0 + 0.05 * self.visits[key])
        self.q[key][i] += decay * delta
        self.weights[i] += self.approximator_lr * delta * x / (1.0 + float(x @ x))
        self.td_errors.append(float(delta))
        if len(self.td_errors) > 512:
            self.td_errors = self.td_errors[-512:]

    def diagnostics(self, state: OrchestratorState) -> dict:
        scores = self.score_actions(state)
        key = state.discretize(self.bins)
        return {
            "model": "Tabular Q + linear approximator",
            "discount": self.discount,
            "state_bucket": list(key),
            "visits": int(self.visits.get(key, 0)),
            "known_states": len(self.q),
            "mean_abs_td_error": float(np.mean(np.abs(self.td_errors))) if self.td_errors else 0.0,
            "q_values": [
                {"action": a.value, "q": float(v)} for a, v in zip(ACTIONS, scores, strict=True)
            ],
        }

    def state_dict(self) -> dict:
        payload = super().state_dict()
        payload.update(
            {
                "learning_rate": self.learning_rate,
                "approximator_lr": self.approximator_lr,
                "discount": self.discount,
                "bins": self.bins,
                "optimistic_init": self.optimistic_init,
                "weights": self.weights.tolist(),
                "q": {",".join(map(str, k)): v.tolist() for k, v in self.q.items()},
                "visits": {",".join(map(str, k)): v for k, v in self.visits.items()},
                "td_errors": self.td_errors[-128:],
            }
        )
        return payload

    def load_state_dict(self, payload: dict) -> None:
        super().load_state_dict(payload)
        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        self.approximator_lr = float(payload.get("approximator_lr", self.approximator_lr))
        self.discount = float(payload.get("discount", self.discount))
        self.bins = int(payload.get("bins", self.bins))
        self.optimistic_init = float(payload.get("optimistic_init", self.optimistic_init))
        if "weights" in payload:
            self.weights = np.array(payload["weights"], dtype=float)
        for key, values in payload.get("q", {}).items():
            self.q[tuple(int(p) for p in key.split(","))] = np.array(values, dtype=float)
        for key, value in payload.get("visits", {}).items():
            self.visits[tuple(int(p) for p in key.split(","))] = int(value)
        self.td_errors = [float(v) for v in payload.get("td_errors", [])]
