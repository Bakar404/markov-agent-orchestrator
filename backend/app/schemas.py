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
