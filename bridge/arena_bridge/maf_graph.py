"""The arm patterns as actual Microsoft Agent Framework workflows.

The hand-rolled versions in :mod:`workflows` reimplement what the framework already does, so an
arm named after a framework pattern was really measuring our approximation of it. Here the
routing is the framework's: executors are ``Executor`` subclasses and the shape comes from
``WorkflowBuilder`` edges, one edge construct per pattern.

* sequential uses ``add_chain`` with a closing edge, so the chain cycles
* concurrent uses ``add_fan_out_edges`` and ``add_fan_in_edges``, so the graph picks the coalition
* handoff uses ``add_switch_case_edge_group``, so a nomination is dispatched by the routing table

The arena stays the authority. A workflow cannot act on its own because acting means calling
``ctx.request_info``, which suspends the graph until the arena has approved the action, sampled
the transition and charged for it. The framework decides *who is next*; the arena decides
*whether that is allowed and what it costs*.
"""

# No `from __future__ import annotations` here: the framework resolves handler signatures by
# introspection, and stringised annotations make WorkflowContext[Turn] unresolvable.
from dataclasses import dataclass, field

from agent_framework import (
    Case,
    Default,
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


@dataclass
class Proposal:
    """One branch of a fan-out reporting back. Carries the turn so the fan-in can decide."""

    agent: str
    order: int
    solo_legal: bool
    legal: list[str] = field(default_factory=list)
    last_agents: list[str] = field(default_factory=list)
    hint: str = ""


class Dispatch(Executor):
    """Head of the concurrent pattern. Hands the turn to every branch at once."""

    @handler
    async def fan_out(self, turn: Turn, ctx: WorkflowContext[Turn]) -> None:
        await ctx.send_message(turn)

    @response_handler
    async def resumed(
        self, original_request: ActRequest, response: Turn, ctx: WorkflowContext[Turn]
    ) -> None:
        await ctx.send_message(Turn(response.legal, response.last_agents, response.hint))


class Proposer(Executor):
    """One branch of the fan-out. Reports whether its agent could act on its own."""

    def __init__(self, agent: str, order: int) -> None:
        super().__init__(id=f"propose_{agent}")
        self._agent = agent
        self._order = order

    @handler
    async def propose(self, turn: Turn, ctx: WorkflowContext[Proposal]) -> None:
        await ctx.send_message(
            Proposal(
                agent=self._agent,
                order=self._order,
                solo_legal=SOLO_ACTION[self._agent] in turn.legal,
                legal=turn.legal,
                last_agents=turn.last_agents,
                hint=turn.hint,
            )
        )


class Collect(Executor):
    """Fan-in. Turns the branches back into the single action the arena will price."""

    @handler
    async def gather(self, proposals: list[Proposal], ctx: WorkflowContext[Turn]) -> None:
        # Fan-in makes no ordering promise, and the coalition is part of what is being
        # compared, so the branches are put back in graph order before anything is proposed.
        ordered = sorted(proposals, key=lambda p: p.order)
        legal = ordered[0].legal

        if "run_parallel" in legal:
            await ctx.request_info(
                ActRequest("run_parallel", [p.agent for p in ordered]), Turn
            )
            return
        # Parallel work is gated on remaining budget, so a branch that can still act alone is
        # better than stopping: a concurrent arm out of room still has work it can do.
        solo = next((p.agent for p in ordered if p.solo_legal), None)
        action = ActRequest(SOLO_ACTION[solo], [solo]) if solo else ActRequest("terminate", [])
        await ctx.request_info(action, Turn)

    @response_handler
    async def resumed(
        self, original_request: ActRequest, response: Turn, ctx: WorkflowContext[Turn]
    ) -> None:
        await ctx.send_message(Turn(response.legal, response.last_agents, response.hint))


def build_concurrent(
    coalition: tuple[str, ...] = ("researcher", "critic", "verifier"),
) -> Workflow:
    """Fan out to a coalition, fan back in, and let the arena price the whole step once.

    The membership of the coalition is the graph's shape rather than a list in a policy, which
    is the part worth comparing: ``add_fan_out_edges`` decides who is asked and
    ``add_fan_in_edges`` decides what the arena is finally asked for.
    """
    gate = Gate(id="gate")
    dispatch = Dispatch(id="dispatch")
    branches = [Proposer(agent, order) for order, agent in enumerate(coalition)]
    collect = Collect(id="collect")

    builder = WorkflowBuilder(start_executor=gate, max_iterations=200)
    builder.add_edge(gate, dispatch)
    builder.add_fan_out_edges(dispatch, branches)
    builder.add_fan_in_edges(branches, collect)
    builder.add_edge(collect, dispatch)
    return builder.build()


@dataclass
class Routed:
    """A turn with a nomination stamped on it, so switch-case edges can dispatch it."""

    target: str = ""
    legal: list[str] = field(default_factory=list)
    last_agents: list[str] = field(default_factory=list)
    hint: str = ""


class Router(Executor):
    """Reads the nomination out of the last agent's output and stamps it on the turn.

    The choice is data-dependent, which is the whole point of handoff, so the routing table is
    a switch-case edge group rather than a branch: the graph shows where a nomination can go.
    """

    def __init__(self, id: str) -> None:
        super().__init__(id=id)
        self._seen: list[str] = []

    @handler
    async def route(self, turn: Turn, ctx: WorkflowContext[Routed]) -> None:
        await self._dispatch(turn, ctx)

    @response_handler
    async def resumed(
        self, original_request: ActRequest, response: Turn, ctx: WorkflowContext[Routed]
    ) -> None:
        await self._dispatch(response, ctx)

    async def _dispatch(self, turn: Turn, ctx: WorkflowContext[Routed]) -> None:
        lowered = turn.hint.lower()
        # A nomination that names whoever just acted would loop, so it falls through to whoever
        # has not been used yet, and only then to anyone still legal.
        pools = (
            [s for s in SPECIALISTS if s in lowered and s not in turn.last_agents],
            [s for s in SPECIALISTS if s not in self._seen],
            list(SPECIALISTS),
        )
        for pool in pools:
            nominated = next((s for s in pool if SOLO_ACTION[s] in turn.legal), None)
            if nominated is not None:
                self._seen.append(nominated)
                await ctx.send_message(
                    Routed(nominated, turn.legal, turn.last_agents, turn.hint)
                )
                return
        await ctx.send_message(Routed("", turn.legal, turn.last_agents, turn.hint))


class Nominee(Executor):
    """A switch-case destination. Acts as the agent the router named."""

    def __init__(self, agent: str) -> None:
        super().__init__(id=f"act_{agent}")
        self._agent = agent

    @handler
    async def act(self, routed: Routed, ctx: WorkflowContext[Turn]) -> None:
        await ctx.request_info(ActRequest(SOLO_ACTION[self._agent], [self._agent]), Turn)

    @response_handler
    async def resumed(
        self, original_request: ActRequest, response: Turn, ctx: WorkflowContext[Turn]
    ) -> None:
        await ctx.send_message(Turn(response.legal, response.last_agents, response.hint))


class Halt(Executor):
    """The default branch: nobody legal was nominated, so propose stopping."""

    @handler
    async def stop(self, routed: Routed, ctx: WorkflowContext[Turn]) -> None:
        await ctx.request_info(ActRequest("terminate", []), Turn)

    @response_handler
    async def resumed(
        self, original_request: ActRequest, response: Turn, ctx: WorkflowContext[Turn]
    ) -> None:
        await ctx.send_message(Turn(response.legal, response.last_agents, response.hint))


def build_handoff() -> Workflow:
    """Agent-directed routing, dispatched through a switch-case edge group."""
    gate = Gate(id="gate")
    router = Router(id="router")
    nominees = {agent: Nominee(agent) for agent in SPECIALISTS}
    halt = Halt(id="halt")

    def names(agent: str):
        return lambda message: getattr(message, "target", "") == agent

    builder = WorkflowBuilder(start_executor=gate, max_iterations=200)
    builder.add_edge(gate, router)
    builder.add_switch_case_edge_group(
        router,
        [Case(names(agent), node) for agent, node in nominees.items()] + [Default(halt)],
    )
    for node in nominees.values():
        builder.add_edge(node, router)
    builder.add_edge(halt, router)
    return builder.build()


GRAPHS = {
    "maf_sequential": build_sequential,
    "maf_concurrent": build_concurrent,
    "maf_handoff": build_handoff,
}
