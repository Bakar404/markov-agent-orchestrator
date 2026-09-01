"""Cooperative Markov game orchestration — the only policy here that learns.

The orchestrator stops treating "which agent" as a flat action and starts treating the agents
as *players* in a cooperative stochastic game (Shapley, 1953; Littman, 1994). Each player has
its own value function over the shared state, and a learned pairwise synergy matrix captures
the fact that a coalition is not the sum of its parts — a Researcher and a Verifier together are
worth more than either alone, while two overlapping agents are worth less.

    V(s, C) = Σ_{i∈C} Q_i(s) + Σ_{i<j ∈ C} W_ij - λ·(|C| - 1)·cost_pressure(s)

The action score for ``RUN_PARALLEL`` is the best multi-player coalition value, so the policy
chooses coalition size endogenously instead of being told when to fan out.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..actions import ACTIONS, ACTION_INDEX, AGENT_TO_ACTION, Action
from ..agents import AGENT_IDS
from ..state import OrchestratorState
from .base import Policy

MAX_COALITION_SIZE = 3


class CooperativeMarkovGamePolicy(Policy):
    id = "markov_game"
    label = "Cooperative Markov Game"
    stage = 3
    family = "markov_game"
    description = (
        "Agents are players in a cooperative stochastic game. Per-player value functions plus a "
        "learned pairwise synergy matrix score every coalition; coalition size is chosen by the "
        "policy, not hard-coded."
    )

    def __init__(
        self,
        *,
        feature_dim: int,
        temperature: float = 0.26,
        learning_rate: float = 0.06,
        synergy_lr: float = 0.03,
        discount: float = 0.93,
        coalition_cost: float = 0.35,
        **kwargs: object,
    ) -> None:
        super().__init__(feature_dim=feature_dim, temperature=temperature, **kwargs)
        self.learning_rate = float(learning_rate)
        self.synergy_lr = float(synergy_lr)
        self.discount = float(discount)
        self.coalition_cost = float(coalition_cost)
        self.agent_ids = list(AGENT_IDS)
        n = len(self.agent_ids)
        self.q_weights = np.zeros((n, feature_dim), dtype=float)
        self.synergy = np.zeros((n, n), dtype=float)
        self.terminate_weights = np.zeros(feature_dim, dtype=float)
        self.coalition_counts: dict[str, int] = {}

    # ------------------------------------------------------------- valuation
    def _player_values(self, state: OrchestratorState) -> np.ndarray:
        return self.q_weights @ state.features()

    def _cost_pressure(self, state: OrchestratorState) -> float:
        return float(1.0 - 0.5 * state.budget_remaining - 0.5 * state.latency_remaining)

    def coalition_value(
        self, state: OrchestratorState, indices: tuple[int, ...], values: np.ndarray | None = None
    ) -> float:
        values = self._player_values(state) if values is None else values
        total = float(values[list(indices)].sum())
        for i, j in combinations(indices, 2):
            total += float(self.synergy[i, j])
        total -= self.coalition_cost * (len(indices) - 1) * self._cost_pressure(state)
        return total

    def _all_coalitions(self) -> list[tuple[int, ...]]:
        n = len(self.agent_ids)
        coalitions: list[tuple[int, ...]] = []
        for size in range(1, min(MAX_COALITION_SIZE, n) + 1):
            coalitions.extend(combinations(range(n), size))
        return coalitions

    def _best_multi_coalition(self, state: OrchestratorState) -> tuple[tuple[int, ...], float]:
        values = self._player_values(state)
        best: tuple[int, ...] = (0, 1)
        best_value = -np.inf
        for coalition in self._all_coalitions():
            if len(coalition) < 2:
                continue
            value = self.coalition_value(state, coalition, values)
            if value > best_value:
                best, best_value = coalition, value
        return best, float(best_value)

    # -------------------------------------------------------------- policy
    def score_actions(self, state: OrchestratorState) -> np.ndarray:
        values = self._player_values(state)
        scores = np.zeros(len(ACTIONS), dtype=float)
        for idx, agent_id in enumerate(self.agent_ids):
            scores[ACTION_INDEX[AGENT_TO_ACTION[agent_id]]] = self.coalition_value(
                state, (idx,), values
            )
        _, parallel_value = self._best_multi_coalition(state)
        scores[ACTION_INDEX[Action.RUN_PARALLEL]] = parallel_value
        scores[ACTION_INDEX[Action.TERMINATE]] = float(
            self.terminate_weights @ state.features()
        ) + self._termination_readiness(state)
        return scores

    @staticmethod
    def _termination_readiness(state: OrchestratorState) -> float:
        """Grounded prior so the game does not have to learn 'do not quit at step 1'."""
        ready = state.confidence * state.verification_score * (1.0 - state.unresolved_ratio)
        return float(-1.6 + 3.4 * ready + 1.8 * (1.0 - state.budget_remaining))

    def preferred_coalition(self, state: OrchestratorState) -> list[str]:
        indices, _ = self._best_multi_coalition(state)
        return [self.agent_ids[i] for i in indices]

    def agent_preferences(self, state: OrchestratorState) -> dict[str, float]:
        values = self._player_values(state)
        shifted = values - values.min() + 1e-3
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

        if action is Action.TERMINATE or not agents:
            target = reward
            current = float(self.terminate_weights @ x) + self._termination_readiness(state)
            self.terminate_weights += self.learning_rate * (target - current) * x / norm
            return

        indices = tuple(self.agent_ids.index(a) for a in agents if a in self.agent_ids)
        if not indices:
            return

        key = "+".join(sorted(agents))
        self.coalition_counts[key] = self.coalition_counts.get(key, 0) + 1

        bootstrap = 0.0
        if not done:
            next_values = self._player_values(next_state)
            bootstrap = max(
                self.coalition_value(next_state, c, next_values) for c in self._all_coalitions()
            )
        target = reward + self.discount * bootstrap

        # Individual credit: the difference reward from the reward model, bootstrapped.
        for agent_id in agents:
            i = self.agent_ids.index(agent_id)
            individual_target = per_agent_reward.get(agent_id, reward / len(agents))
            individual_target += self.discount * bootstrap / len(agents)
            delta = individual_target - float(self.q_weights[i] @ x)
            self.q_weights[i] += self.learning_rate * delta * x / norm

        # Synergy absorbs whatever the additive decomposition cannot explain.
        predicted = self.coalition_value(state, indices)
        residual = target - predicted
        pairs = list(combinations(indices, 2))
        if pairs:
            share = self.synergy_lr * residual / len(pairs)
            for i, j in pairs:
                self.synergy[i, j] += share
                self.synergy[j, i] = self.synergy[i, j]

    def diagnostics(self, state: OrchestratorState) -> dict:
        values = self._player_values(state)
        coalition, value = self._best_multi_coalition(state)
        return {
            "model": "Cooperative Markov game with pairwise synergy",
            "discount": self.discount,
            "cost_pressure": self._cost_pressure(state),
            "player_values": [
                {"agent": agent_id, "value": float(values[i])}
                for i, agent_id in enumerate(self.agent_ids)
            ],
            "best_coalition": {
                "agents": [self.agent_ids[i] for i in coalition],
                "value": value,
            },
            "top_synergies": self._top_synergies(),
            "coalition_counts": dict(sorted(self.coalition_counts.items(), key=lambda kv: -kv[1])[:6]),
        }

    def _top_synergies(self, limit: int = 5) -> list[dict]:
        pairs = []
        for i, j in combinations(range(len(self.agent_ids)), 2):
            pairs.append(
                {
                    "pair": [self.agent_ids[i], self.agent_ids[j]],
                    "synergy": float(self.synergy[i, j]),
                }
            )
        pairs.sort(key=lambda p: -abs(p["synergy"]))
        return pairs[:limit]

    def state_dict(self) -> dict:
        payload = super().state_dict()
        payload.update(
            {
                "learning_rate": self.learning_rate,
                "synergy_lr": self.synergy_lr,
                "discount": self.discount,
                "coalition_cost": self.coalition_cost,
                "q_weights": self.q_weights.tolist(),
                "synergy": self.synergy.tolist(),
                "terminate_weights": self.terminate_weights.tolist(),
                "coalition_counts": self.coalition_counts,
            }
        )
        return payload

    def load_state_dict(self, payload: dict) -> None:
        super().load_state_dict(payload)
        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        self.synergy_lr = float(payload.get("synergy_lr", self.synergy_lr))
        self.discount = float(payload.get("discount", self.discount))
        self.coalition_cost = float(payload.get("coalition_cost", self.coalition_cost))
        if "q_weights" in payload:
            self.q_weights = np.array(payload["q_weights"], dtype=float)
        if "synergy" in payload:
            self.synergy = np.array(payload["synergy"], dtype=float)
        if "terminate_weights" in payload:
            self.terminate_weights = np.array(payload["terminate_weights"], dtype=float)
        self.coalition_counts = dict(payload.get("coalition_counts", {}))
