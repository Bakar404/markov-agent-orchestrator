"""The action space A of the orchestration MDP / Markov game."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    INVOKE_PLANNER = "invoke_planner"
    INVOKE_RESEARCHER = "invoke_researcher"
    INVOKE_CRITIC = "invoke_critic"
    INVOKE_VERIFIER = "invoke_verifier"
    INVOKE_EXECUTOR = "invoke_executor"
    INVOKE_MEMORY = "invoke_memory"
    RUN_PARALLEL = "run_parallel"
    TERMINATE = "terminate"


ACTIONS: tuple[Action, ...] = tuple(Action)
ACTION_INDEX: dict[Action, int] = {a: i for i, a in enumerate(ACTIONS)}

SINGLE_AGENT_ACTIONS: dict[Action, str] = {
    Action.INVOKE_PLANNER: "planner",
    Action.INVOKE_RESEARCHER: "researcher",
    Action.INVOKE_CRITIC: "critic",
    Action.INVOKE_VERIFIER: "verifier",
    Action.INVOKE_EXECUTOR: "executor",
    Action.INVOKE_MEMORY: "memory",
}

AGENT_TO_ACTION: dict[str, Action] = {v: k for k, v in SINGLE_AGENT_ACTIONS.items()}


@dataclass(frozen=True)
class ActionSpec:
    action: Action
    label: str
    description: str
    kind: str


ACTION_SPECS: dict[Action, ActionSpec] = {
    Action.INVOKE_PLANNER: ActionSpec(
        Action.INVOKE_PLANNER, "Invoke Planner", "Decompose and re-plan the frontier.", "agent"
    ),
    Action.INVOKE_RESEARCHER: ActionSpec(
        Action.INVOKE_RESEARCHER,
        "Invoke Researcher",
        "Acquire external evidence; highest expected information gain.",
        "agent",
    ),
    Action.INVOKE_CRITIC: ActionSpec(
        Action.INVOKE_CRITIC,
        "Invoke Critic",
        "Adversarially review the leading hypothesis.",
        "agent",
    ),
    Action.INVOKE_VERIFIER: ActionSpec(
        Action.INVOKE_VERIFIER,
        "Invoke Verifier",
        "Check the artifact against constraints and citations.",
        "agent",
    ),
    Action.INVOKE_EXECUTOR: ActionSpec(
        Action.INVOKE_EXECUTOR, "Invoke Executor", "Produce or mutate the artifact.", "agent"
    ),
    Action.INVOKE_MEMORY: ActionSpec(
        Action.INVOKE_MEMORY,
        "Memory Retrieval",
        "Retrieve prior context; cheap deduplication.",
        "agent",
    ),
    Action.RUN_PARALLEL: ActionSpec(
        Action.RUN_PARALLEL,
        "Run Parallel",
        "Fan out to a coalition of agents; costs add, latency is the max.",
        "coalition",
    ),
    Action.TERMINATE: ActionSpec(
        Action.TERMINATE, "Terminate", "Stop and emit the terminal reward.", "control"
    ),
}


def action_catalog() -> list[dict]:
    return [
        {
            "id": spec.action.value,
            "label": spec.label,
            "description": spec.description,
            "kind": spec.kind,
            "agent": SINGLE_AGENT_ACTIONS.get(spec.action),
        }
        for spec in ACTION_SPECS.values()
    ]
