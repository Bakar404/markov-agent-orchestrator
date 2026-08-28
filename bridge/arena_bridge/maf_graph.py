"""The arm patterns as actual Microsoft Agent Framework workflows.

The hand-rolled versions in :mod:`workflows` reimplement what the framework already does, so an
arm named after a framework pattern was really measuring our approximation of it. Here the
routing is the framework's: executors are ``Executor`` subclasses, the order comes from
``WorkflowBuilder`` edges, and a cycle edge is what makes the chain repeat.

The arena stays the authority. A workflow cannot act on its own because acting means calling
``ctx.request_info``, which suspends the graph until the arena has approved the action, sampled
the transition and charged for it. The framework decides *who is next*; the arena decides
*whether that is allowed and what it costs*.
"""

# No `from __future__ import annotations` here: the framework resolves handler signatures by
# introspection, and stringised annotations make WorkflowContext[Turn] unresolvable.
from dataclasses import dataclass, field

from agent_framework import (
    Executor,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    handler,
    response_handler,
)

SPECIALISTS = ("planner", "researcher", "critic", "verifier", "memory", "executor")
SOLO_ACTION = {agent: f"invoke_{agent}" for agent in (*SPECIALISTS, "generalist")}


@dataclass
class Turn:
    """What the arena currently permits. Travels the graph as the message type."""

    legal: list[str] = field(default_factory=list)
    last_agents: list[str] = field(default_factory=list)
    hint: str = ""
    idle_hops: int = 0
    """Consecutive nodes that could not act.

    A cycle with nothing legal left would spin until the iteration cap, so a full lap of
    ineligible nodes is what tells the graph to propose termination instead."""


@dataclass
class ActRequest:
    """A proposal put to the arena. Suspends the workflow until answered."""

    action: str
    agents: list[str]


class Gate(Executor):
    """Works solo until the arena makes escalation legal, then takes it.

    Specialists are earned rather than assumed, so this node exists in every pattern and pays
    the same entry cost. Modelling it as a graph node rather than a branch in the adapter keeps
    the claim honest: the opening really is part of the workflow.
    """

    @handler
    async def start(self, turn: Turn, ctx: WorkflowContext[Turn]) -> None:
        await self._advance(turn, ctx)

    @response_handler
    async def resumed(
        self, original_request: ActRequest, response: Turn, ctx: WorkflowContext[Turn]
    ) -> None:
        await self._advance(response, ctx)

    async def _advance(self, turn: Turn, ctx: WorkflowContext[Turn]) -> None:
        if "escalate" in turn.legal:
            await ctx.request_info(ActRequest("escalate", []), Turn)
        elif "invoke_generalist" in turn.legal:
            await ctx.request_info(ActRequest("invoke_generalist", ["generalist"]), Turn)
        else:
            # The roster is open, so routing becomes the pattern's job.
            await ctx.send_message(Turn(turn.legal, turn.last_agents, turn.hint))


class Specialist(Executor):
    """One node in the pattern. Acts when the arena allows it, otherwise passes the turn on."""

    def __init__(self, agent: str, lap: int) -> None:
        super().__init__(id=f"{agent}_{lap}")
        self._agent = agent
        self._lap = lap

    @handler
    async def act(self, turn: Turn, ctx: WorkflowContext[Turn]) -> None:
        if turn.idle_hops >= self._lap:
            await ctx.request_info(ActRequest("terminate", []), Turn)
            return
        if SOLO_ACTION[self._agent] in turn.legal:
            await ctx.request_info(ActRequest(SOLO_ACTION[self._agent], [self._agent]), Turn)
            return
        await ctx.send_message(
            Turn(turn.legal, turn.last_agents, turn.hint, turn.idle_hops + 1)
        )

    @response_handler
    async def resumed(
        self, original_request: ActRequest, response: Turn, ctx: WorkflowContext[Turn]
    ) -> None:
        await ctx.send_message(Turn(response.legal, response.last_agents, response.hint))


def build_sequential(order: tuple[str, ...] = ("planner", "researcher", "critic", "verifier")) -> Workflow:
    """A fixed chain that repeats, expressed as real framework edges.

    ``add_chain`` lays down the order and a closing edge back to the head turns it into a cycle,
    which is what lets one pattern cover a run of any length.
    """
    gate = Gate(id="gate")
    nodes = [Specialist(agent, lap=len(order)) for agent in order]

    builder = WorkflowBuilder(start_executor=gate, max_iterations=200)
    builder.add_edge(gate, nodes[0])
    builder.add_chain(nodes)
    builder.add_edge(nodes[-1], nodes[0])
    return builder.build()


GRAPHS = {"maf_sequential": build_sequential}
