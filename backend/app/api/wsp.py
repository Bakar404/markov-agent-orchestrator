"""Weight-space programming: what an editor is shown before it writes a patch."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..orchestration.engine import RunConfig
from ..orchestration.wsp.calibrate import calibration_overrides
from ..orchestration.wsp.program import DEFAULT_GOAL, brief

router = APIRouter(prefix="/api/wsp", tags=["wsp"])


@router.get("/brief")
def wsp_brief(
    goal: str = Query(DEFAULT_GOAL, min_length=3, max_length=2000),
    task_complexity: float = Query(0.55, ge=0.05, le=0.99),
    max_steps: int = Query(40, ge=1, le=500),
    belief_dim: int = Query(8, ge=3, le=32),
    budget_usd: float = Query(1.20, gt=0.0, le=20_000_000.0),
    latency_budget_ms: float = Query(90_000.0, gt=0.0, le=6_000_000.0),
    stochasticity: float = Query(1.0, ge=0.05, le=3.0),
    confidence_target: float = Query(0.55, ge=0.1, le=0.999),
    verification_target: float = Query(0.75, ge=0.0, le=1.0),
    min_steps_before_terminate: int = Query(3, ge=0, le=50),
    needs_evidence: float | None = Query(None, ge=0.0, le=1.0),
    needs_execution: float | None = Query(None, ge=0.0, le=1.0),
    needs_verification: float | None = Query(None, ge=0.0, le=1.0),
) -> dict:
    """Rendered weights, calibration diagnostics and edit limits for the base policy.

    An out-of-process editor (the Copilot bridge) reads this, proposes a patch, and submits it
    as ``wsp_patch`` when creating a ``wsp_markov_game`` run, where the guard vets it. Only
    calibration seeds are ever run here, under the same task settings the run will have — pass
    the run's budget, latency budget and task shape, or the editor sees a different episode.
    """
    shape = {
        name: value
        for name, value in (
            ("needs_evidence", needs_evidence),
            ("needs_execution", needs_execution),
            ("needs_verification", needs_verification),
        )
        if value is not None
    }
    try:
        config = RunConfig(
            task="wsp brief",
            policy="markov_game",
            task_complexity=task_complexity,
            budget_usd=budget_usd,
            latency_budget_ms=latency_budget_ms,
            max_steps=max_steps,
            belief_dim=belief_dim,
            stochasticity=stochasticity,
            confidence_target=confidence_target,
            verification_target=verification_target,
            min_steps_before_terminate=min_steps_before_terminate,
            task_shape=shape,
        )
        result = brief(goal, **calibration_overrides(config.to_dict()))
    except (ValueError, OverflowError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {**result.to_dict(), "default_goal": DEFAULT_GOAL}
