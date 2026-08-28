"""Does each framework graph route the same way its hand-rolled version did?

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

SPECIALISTS = ("planner", "researcher", "critic", "verifier", "memory", "executor")

PAIRS = (
    ("hand_rolled_sequential", "maf_sequential"),
    ("hand_rolled_concurrent", "maf_concurrent"),
    ("hand_rolled_handoff", "maf_handoff"),
)

# The handoff pattern reads a nomination out of the last agent's text, so the scripted arena
# has to feed one. Anything else would only exercise its fallback.
HINTS = ("", "", "critic", "verifier", "planner", "", "researcher", "", "", "")


def legal_for(step: int, *, escalated: bool, roster: bool, parallel: bool) -> list[str]:
    """A stand-in arena: solo, then escalation offered, then the roster."""
    if step == 0:
        return ["invoke_generalist"]
    if not escalated:
        return ["invoke_generalist", "escalate"]
    if not roster:
        return ["terminate"]
    actions = [f"invoke_{a}" for a in SPECIALISTS] + ["terminate"]
    return [*actions, "run_parallel"] if parallel else actions


async def trace(arm: str, *, steps: int, roster: bool = True, parallel: bool = True) -> list[str]:
    workflow = build_workflow(arm)
    assert workflow is not None, f"no workflow for {arm}"

    seen: list[str] = []
    escalated = False
    last_agents: list[str] = []
    for step in range(steps):
        legal = legal_for(step, escalated=escalated, roster=roster, parallel=parallel)
        hint = HINTS[step] if step < len(HINTS) else ""
        choice = await workflow.choose(legal=legal, last_agents=last_agents, hint=hint)
        rendered = choice.action
        if len(choice.agents) > 1:
            rendered = f"{choice.action}({'+'.join(choice.agents)})"
        seen.append(rendered)
        if choice.action == "escalate":
            escalated = True
        if choice.action == "terminate":
            break
        last_agents = choice.agents
    return seen


async def compare(hand_arm: str, maf_arm: str, *, label: str = "", **kwargs) -> bool:
    hand = await trace(hand_arm, **kwargs)
    maf = await trace(maf_arm, **kwargs)
    width = max((len(a) for a in hand + maf), default=10) + 2

    print(f"{maf_arm}{f'  [{label}]' if label else ''}")
    print(f"  {'':>2} {hand_arm:<{width + 4}}{maf_arm}")
    for i in range(max(len(hand), len(maf))):
        a = hand[i] if i < len(hand) else ""
        b = maf[i] if i < len(maf) else ""
        flag = "" if a == b else "   <-- differs"
        print(f"  {i:>2} {a:<{width + 4}}{b}{flag}")
    ok = hand == maf
    print(f"  => {'MATCH' if ok else 'DIFFERENT'}\n")
    return ok


async def main() -> int:
    results = []
    for hand_arm, maf_arm in PAIRS:
        results.append(await compare(hand_arm, maf_arm, label="full roster", steps=10))

    # Nothing the pattern wants is legal, so both should give up rather than spin.
    for hand_arm, maf_arm in PAIRS:
        results.append(
            await compare(hand_arm, maf_arm, label="nothing legal", steps=6, roster=False)
        )

    # Budget gone, so the concurrent arm cannot fan out and has to fall back to acting alone.
    results.append(
        await compare(
            "hand_rolled_concurrent",
            "maf_concurrent",
            label="no parallel budget",
            steps=6,
            parallel=False,
        )
    )

    if all(results):
        print("ALL MATCH - every framework graph routes like its hand-rolled reference")
        return 0
    print(f"{results.count(False)} of {len(results)} scenarios differ - investigate")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
