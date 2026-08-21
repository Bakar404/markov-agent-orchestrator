"""Policy interface shared by every decision model.

A policy maps the state to a *distribution* over legal actions. The orchestrator always samples
from that distribution rather than taking the argmax, so exploration is visible in the UI and
the recorded ``action_probability`` is the true probability of the branch that was taken.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..actions import ACTIONS, Action
from ..state import OrchestratorState


def masked_softmax(
    scores: np.ndarray, mask: np.ndarray, temperature: float = 1.0
) -> np.ndarray:
    """Boltzmann distribution over the legal actions only."""
    temperature = max(float(temperature), 1e-3)
    z = np.where(mask, scores / temperature, -np.inf)
    if not np.any(np.isfinite(z)):
        uniform = mask.astype(float)
        return uniform / max(uniform.sum(), 1e-9)
    z = z - np.max(z[np.isfinite(z)])
    exp = np.where(np.isfinite(z), np.exp(z), 0.0)
    total = exp.sum()
    if total <= 1e-12:
        uniform = mask.astype(float)
        return uniform / max(uniform.sum(), 1e-9)
    return exp / total


class Policy(ABC):
    """Base class for all orchestration policies."""

    id: str = "base"
    label: str = "Base Policy"
    stage: int = 0
    family: str = "baseline"
    description: str = ""

    def __init__(self, *, feature_dim: int, temperature: float = 1.0, **_: object) -> None:
        self.feature_dim = feature_dim
        self.temperature = temperature

    # ------------------------------------------------------------- interface
    @abstractmethod
    def score_actions(self, state: OrchestratorState) -> np.ndarray:
        """Unnormalized preference for every action in ``ACTIONS`` order."""

    def distribution(self, state: OrchestratorState, legal: list[Action]) -> np.ndarray:
        mask = np.array([a in legal for a in ACTIONS], dtype=bool)
        return masked_softmax(self.score_actions(state), mask, self.temperature)

    def select(
        self, state: OrchestratorState, legal: list[Action], rng: np.random.Generator
    ) -> tuple[Action, float, dict[str, float]]:
        probabilities = self.distribution(state, legal)
        index = int(rng.choice(len(ACTIONS), p=probabilities))
        action = ACTIONS[index]
        as_dict = {a.value: float(p) for a, p in zip(ACTIONS, probabilities, strict=True)}
        return action, float(probabilities[index]), as_dict

    def agent_preferences(self, state: OrchestratorState) -> dict[str, float] | None:
        """Optional coalition scores used to resolve ``RUN_PARALLEL``."""
        return None

    def preferred_coalition(self, state: OrchestratorState) -> list[str] | None:
        """Explicit coalition for ``RUN_PARALLEL``; game-theoretic policies override this."""
        return None

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
        """Learning hook. Baselines leave this empty."""

    # --------------------------------------------------------- serialization
    def state_dict(self) -> dict:
        return {"id": self.id, "temperature": self.temperature}

    def load_state_dict(self, payload: dict) -> None:
        self.temperature = float(payload.get("temperature", self.temperature))

    def metadata(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "stage": self.stage,
            "family": self.family,
            "description": self.description,
        }

    def diagnostics(self, state: OrchestratorState) -> dict:
        """Policy-specific numbers surfaced in the UI's decision inspector."""
        return {}
