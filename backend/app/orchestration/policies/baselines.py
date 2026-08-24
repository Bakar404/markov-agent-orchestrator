"""Stage 0 baselines: uniform sampling and a hand-tuned need-based heuristic."""

from __future__ import annotations

import numpy as np

from ..actions import ACTIONS, Action
from ..state import OrchestratorState
from .base import Policy


class RandomPolicy(Policy):
    id = "random"
    label = "Uniform Random"
    stage = 0
    family = "baseline"
    description = (
        "Samples uniformly from the legal action set. The control condition every other "
        "policy is measured against."
    )

    def score_actions(self, state: OrchestratorState) -> np.ndarray:
        return np.zeros(len(ACTIONS), dtype=float)


class HeuristicPolicy(Policy):
    id = "heuristic"
    label = "Need-Based Heuristic"
    stage = 0
    family = "baseline"
    description = (
        "Hand-written scoring over the state features: research when uncertain, criticize when "
        "quality lags, verify before terminating. No learning."
    )

    def __init__(self, *, feature_dim: int, temperature: float = 0.35, **kwargs: object) -> None:
        super().__init__(feature_dim=feature_dim, temperature=temperature, **kwargs)

    def score_actions(self, state: OrchestratorState) -> np.ndarray:
        planned = state.step > 0
        scores = {
            Action.INVOKE_GENERALIST: 1.0,
            Action.INVOKE_PLANNER: 1.4 if not planned else 0.25 + 0.5 * state.unresolved_ratio,
            Action.INVOKE_RESEARCHER: 0.2 + 1.6 * state.uncertainty * state.budget_remaining,
            Action.INVOKE_CRITIC: 0.1 + 1.1 * (1.0 - state.quality) * float(planned),
            Action.INVOKE_VERIFIER: 0.1 + 1.3 * state.quality * (1.0 - state.verification_score),
            Action.INVOKE_EXECUTOR: 0.1 + 1.5 * state.unresolved_ratio * state.confidence,
            Action.INVOKE_MEMORY: 0.15 + 1.0 * (1.0 - state.memory_coverage) + state.duplicate_pressure,
            Action.RUN_PARALLEL: 0.1 + 0.9 * state.unresolved_ratio * state.budget_remaining,
            # Escalate when the solo attempt has stalled or the task is visibly large.
            Action.ESCALATE: (
                -0.6
                + 2.2 * state.stall
                + 1.4 * state.unresolved_ratio * state.task_complexity
            ),
            Action.TERMINATE: (
                -1.5
                + 3.0 * state.confidence * state.verification_score * (1.0 - state.unresolved_ratio)
                + 2.0 * (1.0 - state.budget_remaining)
            ),
        }
        return np.array([scores[a] for a in ACTIONS], dtype=float)

    def agent_preferences(self, state: OrchestratorState) -> dict[str, float]:
        return {
            "planner": 0.3 + 0.7 * state.unresolved_ratio,
            "researcher": 0.2 + 1.0 * state.uncertainty,
            "critic": 0.2 + 0.8 * (1.0 - state.quality),
            "verifier": 0.2 + 0.9 * (1.0 - state.verification_score),
            "memory": 0.2 + 0.8 * (1.0 - state.memory_coverage),
            "executor": 0.2 + 0.9 * state.unresolved_ratio,
        }
