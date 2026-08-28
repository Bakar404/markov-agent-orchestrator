"""Adapter between the driver's step loop and a suspending Agent Framework workflow.

The driver asks "who acts next?" once per step. A framework workflow instead runs until it needs
something, so the two meet here: each ``choose`` resumes the graph with the arena's latest
answer and hands back the next proposal it suspends on.

Keeping the driver's interface unchanged is deliberate. The comparison is only meaningful if the
framework arms and the arena's own arms travel identical plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_framework import Workflow

from .maf_graph import ActRequest, Turn


@dataclass
class Choice:
    action: str
    agents: list[str]


class GraphWorkflow:
    """Drives a real ``Workflow``, one suspension at a time."""

    def __init__(self, name: str, workflow: Workflow) -> None:
        self.name = name
        self._workflow = workflow
        self._pending_id: str | None = None
        self._started = False

    async def choose(self, *, legal: list[str], last_agents: list[str], hint: str) -> Choice:
        turn = Turn(list(legal), list(last_agents), hint)
        result = (
            await self._workflow.run(responses={self._pending_id: turn})
            if self._started
            else await self._workflow.run(turn)
        )
        self._started = True

        events = result.get_request_info_events()
        if not events:
            # The graph ran out of route before the arena ran out of run. Terminating is the
            # honest answer: inventing a next agent here would be the adapter deciding, which is
            # exactly what this arm is supposed to be measuring.
            self._pending_id = None
            return Choice("terminate", [])

        if len(events) > 1:
            # A graph suspended in several places at once is proposing a fan-out, and one arena
            # step takes one action. Picking one and dropping the rest would quietly change the
            # pattern under test, so refuse until the adapter can express it.
            proposed = ", ".join(e.data.action for e in events)
            raise NotImplementedError(
                f"{self.name} suspended on {len(events)} concurrent requests ({proposed}); "
                "the adapter cannot express a fan-out as a single arena step yet"
            )

        event = events[0]
        self._pending_id = event.request_id
        request: ActRequest = event.data
        return Choice(request.action, list(request.agents))
