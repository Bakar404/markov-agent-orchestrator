"""Strategy catalog — the menu of arms you can put in an experiment.

Each strategy is a policy plus the configuration that makes it a coherent approach, together
with the part of the research taxonomy that motivates it. The library is not decoration here:
picking a strategy is picking a published idea to test against your own baseline.

Strategies deliberately reference a taxonomy category and a search query rather than hardcoding
citations, so the papers shown alongside an arm come from the live library and stay correct as
it grows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Strategy:
    id: str
    label: str
    policy: str
    summary: str
    when: str
    category: str
    """Taxonomy category whose papers motivate this approach."""
    paper_query: str
    is_control: bool = False
    escalates: str = "learned"
    """``never``, ``always``, ``heuristic`` or ``learned`` — how it decides to orchestrate."""
    external_driver: str = ""
    """Names the outside orchestrator, when the arena is not the one choosing agents.

    Worth surfacing rather than burying in the policy id: a reader comparing arms should be able
    to see which ones the arena decided and which ones it only recorded."""
    policy_options: dict | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["policy_options"] = self.policy_options or {}
        return payload


STRATEGIES: tuple[Strategy, ...] = (
    Strategy(
        id="control",
        label="Single Agent (control)",
        policy="single_agent",
        summary="One generalist handles the whole task. Never decomposes, never delegates.",
        when=(
            "Always include this. Without it you cannot attribute any difference to "
            "orchestration rather than to the task, the model or the day."
        ),
        category="Agent Routing",
        paper_query="single agent baseline language model",
        is_control=True,
        escalates="never",
        policy_options={"agent_id": "generalist"},
    ),
    Strategy(
        id="always_orchestrate",
        label="Fixed Pipeline",
        policy="fixed_sequence",
        summary=(
            "Escalates immediately, then walks a hardcoded rotation of specialists. "
            "No learning and no adaptation."
        ),
        when=(
            "The upper bookend on orchestration cost, and a surprisingly strong baseline: "
            "in simulation it outscores every learned policy here."
        ),
        category="Agent Orchestration",
        paper_query="multi agent LLM pipeline role decomposition",
        escalates="always",
    ),
    Strategy(
        id="cascade",
        label="Stall-Triggered Cascade",
        policy="heuristic",
        summary=(
            "Works solo until progress stalls, then escalates. Hand-tuned thresholds, "
            "no learning."
        ),
        when=(
            "The classic cost-saving shape: cheap attempt first, escalate only on evidence "
            "that it is not working."
        ),
        category="Agent Routing",
        paper_query="cascade cost quality tradeoff routing",
        escalates="heuristic",
    ),
    Strategy(
        id="learned_markov_game",
        label="Cooperative Markov Game",
        policy="markov_game",
        summary=(
            "Agents are players, not action labels. Per-player values plus a learned pairwise "
            "synergy matrix score every coalition, so fan-out width is chosen rather than fixed."
        ),
        when=(
            "When the question is not just which agent but how many. The only arm here that "
            "picks its own coalition instead of being handed one."
        ),
        category="Stochastic Games",
        paper_query="stochastic game cooperative equilibrium",
    ),
    Strategy(
        id="maf_sequential",
        label="MAF Sequential Workflow",
        policy="external",
        summary=(
            "A Microsoft Agent Framework workflow drives a fixed chain: plan, research, "
            "critique, verify. Real WorkflowBuilder edges, with a cycle edge closing the loop."
        ),
        when=(
            "The baseline shape of most agent frameworks. Compare it against the control "
            "before assuming a published pattern beats one good agent."
        ),
        category="Agent Orchestration",
        paper_query="multi agent LLM pipeline role decomposition",
        escalates="always",
        external_driver="microsoft-agent-framework",
    ),
    Strategy(
        id="maf_concurrent",
        label="MAF Concurrent Workflow",
        policy="external",
        summary=(
            "A Microsoft Agent Framework workflow fans out to researcher, critic and verifier "
            "at once and fans back in. Real edge groups decide the coalition."
        ),
        when=(
            "When the subtasks are genuinely independent. Buys wall-clock time and pays for "
            "it in fresh context per specialist."
        ),
        category="Agent Orchestration",
        paper_query="concurrent multi agent fan out aggregation",
        escalates="always",
        external_driver="microsoft-agent-framework",
    ),
    Strategy(
        id="maf_handoff",
        label="MAF Handoff Workflow",
        policy="external",
        summary=(
            "Each agent names who should act next and a Microsoft Agent Framework "
            "switch-case edge group dispatches on the nomination."
        ),
        when=(
            "When the right specialist depends on what the last one found. The most "
            "agent-directed of the three, and the easiest to send in circles."
        ),
        category="Agent Routing",
        paper_query="agent handoff delegation routing language model",
        escalates="always",
        external_driver="microsoft-agent-framework",
    ),
)

STRATEGY_INDEX: dict[str, Strategy] = {s.id: s for s in STRATEGIES}


def strategy_catalog() -> list[dict]:
    return [s.to_dict() for s in STRATEGIES]


def get_strategy(strategy_id: str) -> Strategy:
    try:
        return STRATEGY_INDEX[strategy_id]
    except KeyError as exc:
        known = ", ".join(sorted(STRATEGY_INDEX))
        raise ValueError(f"Unknown strategy '{strategy_id}'. Available: {known}") from exc
