"""Hand-rolled routing patterns, kept as the reference the framework arms are checked against.

These reimplement sequential, concurrent and handoff routing in about a hundred lines. They are
named ``hand_rolled_*`` because that is what they are: an arm called after a framework pattern
while containing none of that framework was measuring this approximation instead of the thing.
The real framework graphs live in :mod:`maf_graph`.

They stay because they are the control for the port. A framework arm that does not reproduce the
behaviour of its hand-rolled twin has changed something other than its plumbing.

The arena's rules still apply, and that is deliberate. Specialists have to be earned: the arena
will not make ``escalate`` legal until the generalist has attempted the work solo, so every
workflow opens by working alone and pays the same entry cost as the built-in orchestration arms.
An arm that skipped that would look cheaper for a reason with nothing to do with its pattern.

Each workflow is therefore handed the legal action list rather than trusted to know it, so a
pattern cannot declare something the arena would refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .maf_adapter import Choice, GraphWorkflow
from .maf_graph import GRAPHS

SPECIALISTS = ("planner", "researcher", "critic", "verifier", "memory", "executor")
SOLO_ACTION = {agent: f"invoke_{agent}" for agent in (*SPECIALISTS, "generalist")}


@dataclass
class Workflow:
    """Decides who acts next from what the arena says is currently allowed."""

    name: str = "workflow"

    async def choose(self, *, legal: list[str], last_agents: list[str], hint: str) -> Choice:
        opening = self._opening(legal)
        return opening if opening is not None else self._pattern(last_agents, hint, legal)

    @staticmethod
    def _opening(legal: list[str]) -> Choice | None:
        """Work solo until the arena allows escalation, then take it."""
        if "escalate" in legal:
            return Choice("escalate", [])
        if "invoke_generalist" in legal:
            return Choice("invoke_generalist", ["generalist"])
        return None

    def _pattern(self, last_agents: list[str], hint: str, legal: list[str]) -> Choice:
        raise NotImplementedError


@dataclass
class Sequential(Workflow):
    """A fixed chain, one specialist per step, each seeing what came before.

    The default shape of most agent frameworks, and the one worth beating before anything more
    elaborate is worth building.
    """

    name: str = "hand_rolled_sequential"
    order: tuple[str, ...] = ("planner", "researcher", "critic", "verifier")
    _index: int = 0

    def _pattern(self, last_agents: list[str], hint: str, legal: list[str]) -> Choice:
        for offset in range(len(self.order)):
            agent = self.order[(self._index + offset) % len(self.order)]
            if SOLO_ACTION[agent] in legal:
                self._index += offset + 1
                return Choice(SOLO_ACTION[agent], [agent])
        return Choice("terminate", [])


@dataclass
class Concurrent(Workflow):
    """Fan out to several specialists at once, then let the arena fold their answers together.

    Buys wall-clock time and pays for it in fresh context per specialist, which is exactly the
    trade the cost column exists to expose.
    """

    name: str = "hand_rolled_concurrent"
    coalition: tuple[str, ...] = ("researcher", "critic", "verifier")

    def _pattern(self, last_agents: list[str], hint: str, legal: list[str]) -> Choice:
        if "run_parallel" in legal:
            return Choice("run_parallel", list(self.coalition))
        # Parallel work is gated on remaining budget, so fall back to one specialist rather than
        # stopping: a concurrent arm that ran out of room still has work it can do.
        for agent in self.coalition:
            if SOLO_ACTION[agent] in legal:
                return Choice(SOLO_ACTION[agent], [agent])
        return Choice("terminate", [])


@dataclass
class Handoff(Workflow):
    """The acting agent names who should go next, so the route is chosen by the agents.

    The most agent-directed of the three and the easiest to send in circles, so a nomination
    that repeats whoever just acted falls through to the next specialist not yet used.
    """

    name: str = "hand_rolled_handoff"
    _seen: list[str] = field(default_factory=list)

    def _pattern(self, last_agents: list[str], hint: str, legal: list[str]) -> Choice:
        lowered = hint.lower()
        candidates = (
            [s for s in SPECIALISTS if s in lowered and s not in last_agents],
            [s for s in SPECIALISTS if s not in self._seen],
            list(SPECIALISTS),
        )
        for pool in candidates:
            nominated = next((s for s in pool if SOLO_ACTION[s] in legal), None)
            if nominated is not None:
                self._seen.append(nominated)
                return Choice(SOLO_ACTION[nominated], [nominated])
        return Choice("terminate", [])


WORKFLOWS: dict[str, type[Workflow]] = {
    "hand_rolled_sequential": Sequential,
    "hand_rolled_concurrent": Concurrent,
    "hand_rolled_handoff": Handoff,
    # Not yet ported to real framework edges, so they still say what they are.
    "maf_concurrent": Concurrent,
    "maf_handoff": Handoff,
}


def build_workflow(arm: str):
    """Return the workflow for an arm, or None when the arena's own policy is driving."""
    graph = GRAPHS.get(arm)
    if graph is not None:
        return GraphWorkflow(arm, graph())
    factory = WORKFLOWS.get(arm)
    return factory() if factory else None
