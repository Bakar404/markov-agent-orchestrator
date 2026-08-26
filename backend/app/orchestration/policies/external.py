"""A policy that records a choice made outside the arena rather than making one.

Every other policy here answers "who should act next?". This one refuses to, because for some
arms the answer belongs to an external orchestrator — a Microsoft Agent Framework workflow, for
instance, whose sequential, concurrent and handoff patterns *are* the thing under test. Wrapping
those in a policy that second-guesses them would measure the wrapper.

The distinction matters enough to be visible in the data. An arm running on this policy did not
have its agents chosen by the arena, and ``action_probability`` is 1.0 because nothing was
sampled. Recording that honestly is the point: a reader can tell which arms were driven from
outside and which were not, instead of having to trust that they were labelled correctly.

What does not change is the rulebook. The declared action still has to be legal, so an external
workflow pays the same escalation cost as every other arm before it can reach the specialists.
Deferring the choice is not the same as exempting it.
"""

from __future__ import annotations

import numpy as np

from ..actions import ACTIONS, Action
from ..state import OrchestratorState
from .base import Policy


class ExternalPolicy(Policy):
    id = "external"
    label = "Externally Driven"
    stage = 0
    family = "baseline"
    description = (
        "Defers every choice to the caller. For arms where an outside orchestrator decides "
        "who acts, so the arena measures that orchestrator rather than a wrapper around it."
    )

    def __init__(self, *, feature_dim: int, **kwargs: object) -> None:
        super().__init__(feature_dim=feature_dim, **kwargs)
        self._declared: tuple[Action, list[str]] | None = None

    def declare(self, action: Action, agent_ids: list[str]) -> None:
        """Take the caller's choice for the next decision. Consumed by the next ``select``."""
        self._declared = (action, list(agent_ids))

    def score_actions(self, state: OrchestratorState) -> np.ndarray:
        # Only reached if something asks for a distribution without a declaration, e.g. the
        # run preview. Flat is the honest answer: this policy has no preferences.
        return np.zeros(len(ACTIONS), dtype=float)

    def select(
        self, state: OrchestratorState, legal: list[Action], rng: np.random.Generator
    ) -> tuple[Action, float, dict[str, float]]:
        if self._declared is None:
            raise ValueError(
                "This run is externally driven, so the caller must declare who acts. "
                "POST live/open with {'action': ..., 'agents': [...]}."
            )
        action, _ = self._declared
        if action not in legal:
            self._declared = None
            raise ValueError(
                f"action '{action.value}' is not legal here. Legal actions: "
                f"{', '.join(a.value for a in legal)}"
            )
        # Nothing was sampled, so the probability of the branch taken is 1.
        return action, 1.0, {a.value: 1.0 if a is action else 0.0 for a in ACTIONS}

    def preferred_coalition(self, state: OrchestratorState) -> list[str] | None:
        if self._declared is None:
            return None
        _, agent_ids = self._declared
        return list(agent_ids) or None

    def consume(self) -> None:
        """Drop the declaration so a stale one cannot silently drive a later step."""
        self._declared = None

    def state_dict(self) -> dict:
        # Deliberately not persisted. A declaration belongs to one open step, and reviving one
        # from a reloaded run would let a step act on a choice made for a different state.
        return {"id": self.id, "temperature": self.temperature}
