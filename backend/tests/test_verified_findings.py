"""Regression tests for the resource-accounting findings in verified/README.md."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from app.orchestration.agents import AGENTS
from app.orchestration.engine import RunConfig
from app.orchestration.live import report_from_response
from app.orchestration.state import initial_state
from app.orchestration.transitions import TransitionModel
from app.schemas import LiveAgentReport


@pytest.mark.parametrize("field", ["latency_ms", "cost_usd", "tokens"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), -1])
def test_invalid_report_measurements_are_rejected(field, value):
    state = initial_state(task_complexity=0.5, budget_usd=1.0, latency_budget_ms=10_000.0,
                          belief_dim=4, rng=np.random.default_rng(0))
    payload = {"agent_id": "generalist", "outcome": "partial", "confidence": 0.5,
               "response": "r", "tokens": 10, "cost_usd": 0.01, "latency_ms": 1.0,
               field: value}
    with pytest.raises(ValueError, match=field):
        LiveAgentReport.model_validate_json(json.dumps(payload))
    with pytest.raises(ValueError, match=f"{field} must be finite and non-negative"):
        report_from_response(AGENTS["generalist"], state, payload, belief_dim=4)


@pytest.mark.parametrize("field", ["escalation_cost_usd", "escalation_latency_ms"])
@pytest.mark.parametrize("value", [-1.0, float("inf"), float("-inf"), float("nan")])
def test_invalid_escalation_resources_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        RunConfig.from_dict({"task": "t", field: value})
    with pytest.raises(ValueError, match=field):
        TransitionModel(belief_dim=4, **{field: value})


@pytest.mark.parametrize("field", ["budget_usd", "latency_budget_ms"])
@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_invalid_resource_budgets_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        RunConfig(task="t", **{field: value})


@pytest.mark.parametrize("field", ["base_cost_usd", "base_latency_ms", "base_tokens"])
@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_agent_resources_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        replace(AGENTS["generalist"], **{field: value})
