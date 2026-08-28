"""Does the framework graph route the same way the hand-rolled version did?

The port is only safe if the two agree, so this drives both against an identical scripted arena
and compares the action sequences. Run with the MAF venv:

    C:\\venvs\\arena-maf\\Scripts\\python bridge/compare_routing.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from arena_bridge.workflows import build_workflow  # noqa: E402

SPECIALISTS = ("planner", "researcher", "critic", "verifier")


def legal_for(step: int, *, escalated: bool, budget_left: int) -> list[str]:
    """A stand-in arena: solo until step 2, escalation offered, then the roster."""
    if step == 0:
        return ["invoke_generalist"]
    if not escalated:
        return ["invoke_generalist", "escalate"]
    if budget_left <= 0:
        return ["terminate"]
    return [f"invoke_{a}" for a in SPECIALISTS] + ["run_parallel", "terminate"]


async def trace(arm: str, steps: int = 10) -> list[str]:
    workflow = build_workflow(arm)
    assert workflow is not None, f"no workflow for {arm}"

    seen: list[str] = []
    escalated = False
    last_agents: list[str] = []
    for step in range(steps):
        legal = legal_for(step, escalated=escalated, budget_left=steps - step)
        choice = await workflow.choose(legal=legal, last_agents=last_agents, hint="")
        seen.append(choice.action)
        if choice.action == "escalate":
            escalated = True
        if choice.action == "terminate":
            break
        last_agents = choice.agents
    return seen


async def main() -> int:
    hand = await trace("hand_rolled_sequential")
    maf = await trace("maf_sequential")

    width = max(len(a) for a in hand + maf) + 2
    print(f"{'hand_rolled_sequential':<{width + 8}}maf_sequential")
    for i in range(max(len(hand), len(maf))):
        a = hand[i] if i < len(hand) else ""
        b = maf[i] if i < len(maf) else ""
        flag = "" if a == b else "   <-- differs"
        print(f"  {i:>2} {a:<{width + 4}}{b}{flag}")

    print()
    if hand == maf:
        print("MATCH - the framework graph routes identically")
        return 0
    print("DIFFERENT - investigate before trusting the arm name")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
