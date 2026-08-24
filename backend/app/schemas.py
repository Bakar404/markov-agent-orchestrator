"""Pydantic request/response models for the public API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunCreate(BaseModel):
    task: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural-language description of the task to orchestrate.",
    )
    policy: str = Field("contextual_bandit", description="Policy id from /api/meta/policies.")
    strategy: str | None = Field(
        None,
        max_length=64,
        description=(
            "Strategy id from /api/meta/strategies. Sets policy, options and arm name, so "
            "an experiment arm can be chosen from the research catalog rather than wired by hand."
        ),
    )
    seed: int | None = Field(None, ge=0, le=2**31 - 1, description="Omit for a random seed.")
    task_complexity: float = Field(0.55, ge=0.05, le=0.99)
    budget_usd: float = Field(1.20, gt=0.0, le=100.0)
    latency_budget_ms: float = Field(90_000.0, gt=0.0, le=6_000_000.0)
    max_steps: int = Field(40, ge=1, le=500)
    belief_dim: int = Field(8, ge=3, le=32)
    stochasticity: float = Field(1.0, ge=0.05, le=3.0)
    confidence_target: float = Field(0.55, ge=0.1, le=0.999)
    verification_target: float = Field(0.75, ge=0.0, le=1.0)
    min_steps_before_terminate: int = Field(
        3, ge=0, le=50, description="Steps that must elapse before TERMINATE becomes legal."
    )
    mode: Literal["sim", "live"] = Field(
        "sim",
        description="'sim' samples outcomes; 'live' waits for real agent invocations via /live.",
    )
    hypotheses: list[str] = Field(
        default_factory=list,
        max_length=32,
        description="Named competing answers, one per belief dimension. Live mode only.",
    )
    task_shape: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "How this task is shaped: needs_evidence, needs_execution, needs_verification, "
            "each 0-1. Lets one router specialize per task type. Defaults to 0.5 each."
        ),
    )
    policy_profile: str | None = Field(
        None,
        max_length=64,
        description="Named learned parameters to load at start and update on termination.",
    )
    experiment: str | None = Field(
        None, max_length=64, description="Groups arms of the same A/B comparison."
    )
    arm: str | None = Field(
        None, max_length=64, description="Which arm this run is, e.g. 'control' or 'marl'."
    )
    policy_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra policy constructor arguments, e.g. {'agent_id': 'researcher'}.",
    )


class PolicyProfileReset(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    policy: str = Field(..., min_length=1, max_length=64)


class VerdictCreate(BaseModel):
    """An external judgment of answer quality, which the reward function cannot supply."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0-1. Only comparable across arms if the same rubric was applied to each.",
    )
    judge: str = Field("human", max_length=64, description="Who scored it.")
    rubric: str = Field("", max_length=2000, description="What was being scored, and how.")
    notes: str = Field("", max_length=4000)


class LiveAgentReport(BaseModel):
    """What one real agent invocation produced."""

    agent_id: str = Field(..., description="Agent the brief was issued to.")
    outcome: Literal["success", "partial", "failure"]
    confidence: float = Field(
        0.5, ge=0.0, le=1.0, description="How strongly the agent backs its claim."
    )
    claimed_hypothesis: int | None = Field(
        None,
        ge=0,
        description="Index of the hypothesis this response argued for. Omit if it argued for none.",
    )
    response: str = Field("", max_length=20000, description="What the agent actually produced.")
    summary: str = Field("", max_length=400)
    tokens: int | None = Field(None, ge=0, description="Measured; estimated from the spec if omitted.")
    latency_ms: float | None = Field(None, ge=0.0)
    cost_usd: float | None = Field(None, ge=0.0)


class LiveReportRequest(BaseModel):
    token: str = Field(..., description="Token returned by /live/open.")
    reports: list[LiveAgentReport] = Field(default_factory=list, max_length=8)
    hypotheses: list[str] = Field(
        default_factory=list,
        max_length=32,
        description="Name the belief slots. Accepted only while they are still unnamed.",
    )


class RunStepRequest(BaseModel):
    steps: int = Field(1, ge=1, le=100, description="How many steps to advance.")


class RunResetRequest(BaseModel):
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    keep_policy_learning: bool = Field(
        False,
        description="Carry the learned policy parameters into the fresh episode.",
    )


class RunStatusUpdate(BaseModel):
    status: Literal["created", "running", "paused", "completed"]


class ResearchSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=400)
    providers: list[str] | None = Field(
        None, description="Provider ids; defaults to arxiv + semantic_scholar + hits_mcp."
    )
    limit: int = Field(10, ge=1, le=50)
    persist: bool = Field(True, description="Write results into the Research Library.")


class ResearchDiscoverRequest(BaseModel):
    categories: list[str] | None = Field(None, description="Taxonomy categories to sweep.")
    providers: list[str] | None = None
    limit_per_query: int = Field(6, ge=1, le=25)


class CampaignRequest(BaseModel):
    """Paired carried-vs-fresh learning experiment."""

    policies: list[str] | None = Field(
        None,
        description="Policy ids to compare. Defaults to the four learning stages.",
    )
    episodes: int = Field(40, ge=2, le=200, description="Episodes per arm, per policy.")
    seed_base: int = Field(1000, ge=0, le=2**31 - 1)
    max_steps: int = Field(40, ge=5, le=200)
    budget_usd: float = Field(1.20, gt=0.0, le=100.0)
    task_complexity: float = Field(0.55, ge=0.05, le=0.99)


class RunSummary(BaseModel):
    id: str
    task: str
    policy: str
    status: str
    seed: int
    step_count: int
    cumulative_reward: float
    total_cost: float
    total_latency_ms: float
    total_tokens: int
    terminated: bool
    termination_reason: str | None
    created_at: str
    updated_at: str
    confidence: float
    entropy: float


class ApiMessage(BaseModel):
    detail: str


class GenericPayload(BaseModel):
    """Loose envelope for endpoints whose shape is driven by the engine."""

    model_config = {"extra": "allow"}

    data: dict[str, Any] | None = None
