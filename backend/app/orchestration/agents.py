"""Agent registry.

Each agent is a first-class decision target for the orchestrator. The parameters below are the
generative model the transition kernel samples from: they determine cost, latency, token draw,
prior competence, and how strongly the agent sharpens (or blurs) the belief distribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AgentSpec:
    id: str
    label: str
    role: str
    description: str
    color: str
    accent: str
    base_cost_usd: float
    base_latency_ms: float
    base_tokens: int
    prior_alpha: float
    prior_beta: float
    evidence_strength: float
    """How much Dirichlet concentration a successful invocation adds to the belief."""
    noise_strength: float
    """Concentration spread over incorrect hypotheses (models hallucination / bias)."""
    quality_gain: float
    verification_gain: float
    memory_gain: float
    subtask_resolution: float
    canvas_position: tuple[float, float]


AGENT_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec(
        id="planner",
        label="Planner Agent",
        role="Decomposition & strategy",
        description=(
            "Decomposes the task into subtasks, estimates difficulty and proposes the next "
            "orchestration frontier. Cheap, fast, and mostly reduces structural uncertainty."
        ),
        color="#9ca3af",
        accent="rgba(156,163,175,0.16)",
        base_cost_usd=0.012,
        base_latency_ms=650.0,
        base_tokens=900,
        prior_alpha=6.0,
        prior_beta=2.0,
        evidence_strength=0.9,
        noise_strength=0.35,
        quality_gain=0.05,
        verification_gain=0.0,
        memory_gain=0.02,
        subtask_resolution=0.0,
        canvas_position=(80.0, 40.0),
    ),
    AgentSpec(
        id="researcher",
        label="Research Agent",
        role="Evidence acquisition",
        description=(
            "Queries external knowledge (arXiv, Semantic Scholar, Papers With Code, MCP tools) "
            "and folds retrieved evidence into the belief. Highest information gain per call."
        ),
        color="#8b5cf6",
        accent="rgba(139,92,246,0.16)",
        base_cost_usd=0.041,
        base_latency_ms=2100.0,
        base_tokens=3400,
        prior_alpha=5.0,
        prior_beta=2.5,
        evidence_strength=2.1,
        noise_strength=0.55,
        quality_gain=0.06,
        verification_gain=0.02,
        memory_gain=0.18,
        subtask_resolution=0.15,
        canvas_position=(430.0, -110.0),
    ),
    AgentSpec(
        id="critic",
        label="Critic Agent",
        role="Adversarial review",
        description=(
            "Attacks the current hypothesis. Raises quality, but can legitimately *increase* "
            "entropy when it uncovers an unconsidered failure mode."
        ),
        color="#60a5fa",
        accent="rgba(96,165,250,0.16)",
        base_cost_usd=0.026,
        base_latency_ms=1250.0,
        base_tokens=2100,
        prior_alpha=4.5,
        prior_beta=3.0,
        evidence_strength=0.75,
        noise_strength=0.95,
        quality_gain=0.14,
        verification_gain=0.05,
        memory_gain=0.03,
        subtask_resolution=0.05,
        canvas_position=(430.0, 190.0),
    ),
    AgentSpec(
        id="verifier",
        label="Verification Agent",
        role="Ground-truth checking",
        description=(
            "Runs checks against constraints and citations. Converts belief mass into "
            "verified confidence; the main driver of the verification reward term."
        ),
        color="#34d399",
        accent="rgba(52,211,153,0.16)",
        base_cost_usd=0.033,
        base_latency_ms=1750.0,
        base_tokens=2600,
        prior_alpha=5.5,
        prior_beta=2.0,
        evidence_strength=1.5,
        noise_strength=0.22,
        quality_gain=0.07,
        verification_gain=0.22,
        memory_gain=0.04,
        subtask_resolution=0.1,
        canvas_position=(790.0, 40.0),
    ),
    AgentSpec(
        id="memory",
        label="Memory Agent",
        role="Retrieval & deduplication",
        description=(
            "Retrieves prior context from the episodic store. Very cheap, raises memory "
            "coverage and suppresses duplicate work penalties."
        ),
        color="#f59e0b",
        accent="rgba(245,158,11,0.16)",
        base_cost_usd=0.004,
        base_latency_ms=280.0,
        base_tokens=450,
        prior_alpha=7.0,
        prior_beta=1.5,
        evidence_strength=0.6,
        noise_strength=0.25,
        quality_gain=0.02,
        verification_gain=0.02,
        memory_gain=0.3,
        subtask_resolution=0.05,
        canvas_position=(80.0, 250.0),
    ),
    AgentSpec(
        id="executor",
        label="Executor Agent",
        role="Tool use & production",
        description=(
            "Performs the actual work: writes code, calls tools, produces the artifact. "
            "Highest cost and latency, and the only agent that meaningfully burns subtasks."
        ),
        color="#ef4444",
        accent="rgba(239,68,68,0.16)",
        base_cost_usd=0.058,
        base_latency_ms=3200.0,
        base_tokens=4800,
        prior_alpha=4.0,
        prior_beta=3.0,
        evidence_strength=1.0,
        noise_strength=0.6,
        quality_gain=0.16,
        verification_gain=0.03,
        memory_gain=0.06,
        subtask_resolution=0.85,
        canvas_position=(790.0, 300.0),
    ),
)

AGENTS: dict[str, AgentSpec] = {spec.id: spec for spec in AGENT_SPECS}
AGENT_IDS: tuple[str, ...] = tuple(spec.id for spec in AGENT_SPECS)


def agent(agent_id: str) -> AgentSpec:
    try:
        return AGENTS[agent_id]
    except KeyError as exc:  # pragma: no cover - guarded by the API layer
        raise ValueError(f"Unknown agent '{agent_id}'") from exc


def agent_catalog() -> list[dict]:
    """Serializable registry used by the frontend to lay out the canvas."""
    catalog: list[dict] = []
    for spec in AGENT_SPECS:
        payload = asdict(spec)
        payload["canvas_position"] = {
            "x": spec.canvas_position[0],
            "y": spec.canvas_position[1],
        }
        catalog.append(payload)
    return catalog
