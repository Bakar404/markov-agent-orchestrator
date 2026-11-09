"""Pydantic request/response models for the public API."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .orchestration.agents import MAX_REPORT_COST, MAX_REPORT_LATENCY_MS, MAX_REPORT_TOKENS

BUDGET_CEILING = {"usd": 100.0, "aiu": 5_000.0, "tokens": 20_000_000.0}
"""Upper bound per unit. Loose enough not to obstruct, tight enough that a unit mix-up trips it."""

BUDGET_FLOOR = {"usd": 0.0, "aiu": 1.0, "tokens": 1_000.0}
"""Below this a budget is almost certainly still denominated in dollars."""

NUMERIC_POLICY_OPTIONS = frozenset(
    {"temperature", "learning_rate", "synergy_lr", "discount", "coalition_cost"}
)
"""Policy constructor arguments that are numbers, so a boolean there is a mistake, not a 1."""


def _require_finite_number(value: Any, where: str) -> None:
    """A real, finite number — not a bool, not a string, not NaN or infinity."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        # ValueError, not TypeError: pydantic turns only ValueError into a 422.
        raise ValueError(f"{where} must be a number")  # noqa: TRY004
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{where} must be finite")


def _reject_nonfinite(value: Any, where: str) -> None:
    """Walk nested containers and refuse any NaN, infinity or float-overflowing int.

    A non-finite number is accepted by the request, stored with the run, and then makes every
    response that serializes the config fail — so it has to be stopped at the door.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        _require_finite_number(value, where)
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{where}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{where}[{index}]")


class WspProvenance(BaseModel):
    """What an inline patch was written against, copied from the brief the editor read.

    A patch holds absolute values, so it only means what its author intended on the weights it
    was written for; the run refuses it if the base no longer hashes the same.
    """

    base_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    calibration_seeds: list[int] = Field(..., min_length=1, max_length=64)

    @field_validator("calibration_seeds", mode="before")
    @classmethod
    def _seeds_are_ints(cls, value: Any) -> Any:
        if isinstance(value, list) and any(
            isinstance(s, bool) or not isinstance(s, int) for s in value
        ):
            raise ValueError("calibration_seeds must be integers")
        return value


class RunCreate(BaseModel):
    task: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural-language description of the task to orchestrate.",
    )
    policy: str = Field("markov_game", description="Policy id from /api/meta/policies.")
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
    budget_usd: float = Field(
        1.20,
        gt=0.0,
        description=(
            "Spending cap, denominated in `cost_unit`. The ceiling scales with the unit: a "
            "budget of 100 is generous in dollars and absurd in tokens."
        ),
    )
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
    experiment: str | None = Field(
        None, max_length=64, description="Groups arms of the same A/B comparison."
    )
    default_model: str = Field(
        "", max_length=64, description="Model every agent uses unless overridden."
    )
    cost_unit: Literal["usd", "tokens", "aiu"] = Field(
        "usd",
        description=(
            "What cost figures are denominated in. Report what you can actually observe: a "
            "framework that exposes only token counts should say 'tokens' rather than pass "
            "them off as dollars. Must be identical across every arm of an experiment."
        ),
    )
    agent_models: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-agent model override, e.g. {'generalist': 'opus', 'planner': 'mini'}. Lets an "
            "arm buy an expensive orchestrator and cheap workers."
        ),
    )
    arm: str | None = Field(
        None, max_length=64, description="Which arm this run is, e.g. 'control' or 'cascade'."
    )
    policy_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra policy constructor arguments, e.g. {'agent_id': 'researcher'}.",
    )
    wsp_goal: str | None = Field(
        None,
        min_length=3,
        max_length=2000,
        description=(
            "Weight-space-programmed strategies only: the goal the editor programs toward. "
            "Omit for the strategy's default goal."
        ),
    )
    wsp_patch: dict[str, Any] | None = Field(
        None,
        description=(
            "Weight-space-programmed strategies only: a patch an editor already proposed "
            "(e.g. the Copilot bridge), in the {goal, rationale, edits} contract. It still "
            "passes through the guard; a file path is never accepted."
        ),
    )
    wsp_provenance: WspProvenance | None = Field(
        None,
        description=(
            "Required with wsp_patch: the brief's base.sha256 and calibration.seeds. The run "
            "is refused if the base checkpoint no longer matches, and both are stored with "
            "the patch."
        ),
    )

    @field_validator("task_shape", mode="before")
    @classmethod
    def _task_shape_is_finite(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for key, item in value.items():
                _require_finite_number(item, f"task_shape.{key}")
        return value

    @field_validator("policy_options")
    @classmethod
    def _policy_options_are_finite(cls, value: dict[str, Any]) -> dict[str, Any]:
        for key in NUMERIC_POLICY_OPTIONS & set(value):
            _require_finite_number(value[key], f"policy_options.{key}")
        _reject_nonfinite(value, "policy_options")
        return value

    @model_validator(mode="after")
    def _budget_fits_the_unit(self) -> RunCreate:
        ceiling = BUDGET_CEILING[self.cost_unit]
        if self.budget_usd > ceiling:
            raise ValueError(
                f"budget {self.budget_usd:,.0f} exceeds the {ceiling:,.0f} ceiling for "
                f"cost_unit '{self.cost_unit}'"
            )
        # The default budget of 1.20 is sane in dollars and instantly exhausted in tokens, which
        # would produce a zero-step run that looks like a result rather than a mistake.
        if self.cost_unit != "usd" and self.budget_usd <= BUDGET_FLOOR[self.cost_unit]:
            raise ValueError(
                f"budget {self.budget_usd:,.2f} is too small to be denominated in "
                f"'{self.cost_unit}' — set one explicitly"
            )
        return self


class VerdictCreate(BaseModel):
    """An external judgment of answer quality, which the reward function cannot supply.

    Absolute scores compress — judges rating one answer at a time drift to the top of the range.
    Prefer ``POST /api/experiments/{name}/pairwise`` when comparing arms; use this for a
    standalone quality record.
    """

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "0-1. Microsoft Foundry's evaluators use a 1-5 Likert scale with a pass threshold "
            "of 3; the equivalent here is 0.0/0.25/0.5/0.75/1.0 with 0.5 as the pass mark. "
            "Only comparable across arms if the same rubric was applied to each."
        ),
    )
    judge: str = Field("human", max_length=64, description="Who scored it.")
    rubric: str = Field("", max_length=2000, description="What was being scored, and how.")
    notes: str = Field(
        "", max_length=4000, description="Why this score. Foundry evaluators return a 'reason'."
    )


class PairwiseCreate(BaseModel):
    """A blind preference between two runs, which discriminates where absolute scores do not."""

    run_a: str = Field(..., max_length=32)
    run_b: str = Field(..., max_length=32)
    winner: Literal["a", "b", "tie"] = Field(
        ...,
        description=(
            "Which answer was better. Record 'tie' honestly rather than forcing a preference — "
            "ties are evidence that the arms are indistinguishable."
        ),
    )
    judge: str = Field("human", max_length=64)
    rubric: str = Field("", max_length=2000, description="Written before seeing the answers.")
    notes: str = Field("", max_length=4000)


class LiveAgentReport(BaseModel):
    """What one real agent invocation produced.

    Every field here is required because the alternative is the engine inventing it. Omitted
    tokens and cost used to fall back to the agent spec's base rates, which produced a cost
    column that was arithmetic over the roster rather than anything a model actually spent.
    If you cannot measure these, you are not running live -- use sim mode, which is honest
    about being sampled.
    """

    agent_id: str = Field(..., description="Agent the brief was issued to.")
    outcome: Literal["success", "partial", "failure"]
    confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="How strongly the agent backs its claim.",
    )
    claimed_hypothesis: int | None = Field(
        None,
        ge=0,
        description="Index of the hypothesis this response argued for. Omit if it argued for none.",
    )
    response: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="What the agent actually produced. Required: a step with no output is not work.",
    )
    summary: str = Field("", max_length=400)
    tokens: int = Field(
        ...,
        ge=0,
        le=MAX_REPORT_TOKENS,
        description="Measured token count for this invocation (at most 10**12).",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        le=MAX_REPORT_LATENCY_MS,
        allow_inf_nan=False,
        description="Measured wall-clock time (at most 1e9 ms).",
    )
    cost_usd: float = Field(
        ...,
        ge=0.0,
        le=MAX_REPORT_COST,
        allow_inf_nan=False,
        description="Measured cost, in the run's cost unit (at most 1e8). Zero is valid; absent is not.",
    )

    @field_validator("confidence", "tokens", "latency_ms", "cost_usd", mode="before")
    @classmethod
    def _numbers_are_not_booleans(cls, value: Any) -> Any:
        # Lax mode would read `true` as 1, which is a type error in the report, not a measurement.
        if isinstance(value, bool):
            raise ValueError("must be a number, not a boolean")  # noqa: TRY004
        return value

    @field_validator("response")
    @classmethod
    def _response_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("response cannot be blank")
        return value


class LiveOpenRequest(BaseModel):
    """Who an external orchestrator chose to act next.

    Only runs on the ``external`` policy accept this. Anywhere else the arena picks, and letting
    the caller override that would turn a policy decision into a caller preference wearing its
    label. The declared action still has to be legal, so an external workflow escalates before
    reaching the specialists like every other arm.
    """

    action: str | None = Field(
        None, max_length=32, description="Action id from /api/meta/actions."
    )
    agents: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Agent ids that will act. One for a solo action, several for run_parallel.",
    )


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
