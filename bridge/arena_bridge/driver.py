"""Drive one arm from creation to termination.

The loop is deliberately dumb: ask the arena who acts, invoke exactly those agents, report
exactly what came back. Every decision that could bias the comparison — which agent, when to
escalate, when to stop — belongs to the policy on the other side of the wire.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .arena import Arena, ArenaError
from .executor import AgentPool, Invocation
from .workflows import build_workflow


@dataclass
class ArmResult:
    arm: str
    seed: int
    run_id: str
    steps: int
    total_tokens: int
    new_tokens: int
    escalated: bool
    terminated_reason: str
    unparsed_reports: int
    final_answer: str
    driver: str
    """``arena`` when the policy chose the agents, or the workflow that chose them instead."""


async def drive_arm(
    arena: Arena,
    *,
    arm: str,
    strategy: str,
    seed: int,
    task: str,
    experiment: str,
    hypotheses: list[str],
    max_steps: int,
    budget: float,
    latency_budget_ms: float,
    default_model: str,
    agent_models: dict[str, str] | None = None,
    on_event=None,
) -> ArmResult:
    def emit(kind: str, **payload: object) -> None:
        if on_event:
            on_event(kind, {"arm": arm, "seed": seed, **payload})

    run = arena.create_run(
        task=task,
        strategy=strategy,
        arm=arm,
        seed=seed,
        experiment=experiment,
        mode="live",
        belief_dim=len(hypotheses),
        hypotheses=hypotheses,
        max_steps=max_steps,
        budget_usd=budget,
        latency_budget_ms=latency_budget_ms,
        default_model=default_model,
        agent_models=agent_models or {},
        cost_unit="tokens",
    )
    run_id = run["id"]
    emit("run_created", run_id=run_id, watch=f"http://localhost:3000/?run={run_id}")

    workflow = build_workflow(arm)
    pool = AgentPool(default_model=default_model)
    total_tokens = 0
    new_tokens = 0
    escalated = False
    unparsed = 0
    last_texts: list[str] = []
    last_agents: list[str] = []
    handoff_hint = ""
    steps = 0
    reason = "max_steps"
    # Every response carries the current preview, so the workflow can pick from what is
    # actually legal without the driver asking separately.
    legal = list(run["preview"]["legal_actions"])

    while steps < max_steps:
        declared = None
        if workflow is not None:
            choice = workflow.choose(legal=legal, last_agents=last_agents, hint=handoff_hint)
            declared = {"action": choice.action, "agents": choice.agents}

        try:
            pending = arena.open_step(run_id, declared)
        except ArenaError as exc:
            # 409 here means the run already terminated, which is a normal way to finish.
            emit("open_refused", detail=str(exc))
            reason = "terminated"
            break

        agent_ids = pending["agents"]
        if pending["action"] == "escalate" or len(agent_ids) > 1:
            escalated = True
        emit("step_open", step=pending["step"], action=pending["action"], agents=agent_ids)

        # A fan-out step runs its agents at once, because wall clock is what the concurrent
        # pattern is buying. Reporting them one after another would price it as a serial chain.
        invocations: list[Invocation] = list(
            await asyncio.gather(
                *(
                    pool.invoke(brief, ask_for_handoff=arm == "maf_handoff")
                    for brief in pending["briefs"]
                )
            )
        )
        for invocation in invocations:
            if not invocation.parsed:
                unparsed += 1
            total_tokens += invocation.total_tokens
            new_tokens += invocation.new_tokens
            emit(
                "agent_done",
                agent=invocation.agent_id,
                model=invocation.model,
                outcome=invocation.outcome,
                confidence=invocation.confidence,
                new_tokens=invocation.new_tokens,
                cached_tokens=invocation.cached_tokens,
                summary=invocation.summary,
                parsed=invocation.parsed,
            )
        reports = [pool.to_report(inv) for inv in invocations]
        reported = arena.report_step(run_id, pending["token"], reports)
        result = reported["step"]
        legal = list(reported["run"]["preview"]["legal_actions"])
        steps += 1
        last_agents = agent_ids
        handoff_hint = " ".join(inv.next_agent for inv in invocations if inv.next_agent)
        if invocations:
            last_texts = [inv.text for inv in invocations if inv.text]

        emit(
            "step_done",
            step=result["step"],
            reward=result["reward"],
            entropy=result["entropy_after"],
            done=result["done"],
        )

        if result["done"]:
            reason = result.get("termination_reason") or "done"
            break

    detail = arena.run_detail(run_id)
    return ArmResult(
        arm=arm,
        seed=seed,
        run_id=run_id,
        steps=steps,
        total_tokens=total_tokens,
        new_tokens=new_tokens,
        escalated=escalated,
        terminated_reason=detail.get("termination_reason") or reason,
        unparsed_reports=unparsed,
        final_answer="\n\n".join(last_texts).strip(),
        driver=workflow.name if workflow else "arena",
    )
