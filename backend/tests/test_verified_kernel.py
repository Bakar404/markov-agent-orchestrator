"""Differential test: the Python rules against the Verus-proved Rust kernel in ``verified/``.

The Rust kernel is proved to meet a specification of three rules: the escalation gate
(``OrchestrationEngine.legal_actions``), budget accounting (``OrchestratorState``'s remaining
and normalized budget and latency, and the spend in the transition kernel) and the WSP weight-
patch guard (``app.orchestration.wsp_guard``). A proof says nothing about the Python, so this
module checks, on random and adversarial inputs, that the Python does exactly what the proved
kernel does. See verified/README.md for what is proved and what is trusted.

The kernel runs as a small CLI (``verified-kernel``, JSON lines on stdin and stdout). Its path
comes from ``VERIFIED_KERNEL_BIN``, or it is built with cargo. Without cargo or Hypothesis these
tests skip; with ``REQUIRE_VERIFIED=1`` (CI, and verified/check.py) they fail instead.

``VERIFIED_PY_MUTANT`` swaps the guard for a deliberately wrong copy; verified/check.py uses it
as the negative control, and every mutant must make this module fail.
"""

from __future__ import annotations

import importlib.util
import json
import math
import numbers
import os
import shutil
import struct
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REQUIRE = os.environ.get("REQUIRE_VERIFIED") == "1"
CRATE = Path(__file__).resolve().parents[2] / "verified"
EXAMPLES = int(os.environ.get("VERIFIED_EXAMPLES", "300"))

try:
    from hypothesis import HealthCheck, example, given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - depends on the environment
    if REQUIRE:
        raise
    pytest.skip("hypothesis is not installed (pip install --group dev)", allow_module_level=True)

from app.orchestration import wsp_guard
from app.orchestration.actions import ACTIONS, Action
from app.orchestration.agents import AGENTS
from app.orchestration.engine import OrchestrationEngine, RunConfig
from app.orchestration.live import report_from_response
from app.orchestration.state import FEATURE_DIM, OrchestratorState, initial_state
from app.orchestration.transitions import TransitionModel

# ─── The Python guard under test, or a planted mutant of it (the negative control) ────────────

PY_MUTANTS: dict[str, tuple[str, str, str]] = {
    # id: (what it breaks, original text, replacement)
    "PM1": ("R4 lets the synergy diagonal through",
            "not indices[k][0] < indices[k][1]", "not indices[k][0] <= indices[k][1]"),
    "PM2": ("quantize rounds with floor(a + 0.5), which double-rounds just below a half",
            "r = f + 1 if a - f >= 0.5 else f", "r = math.floor(a + 0.5)"),
    "PM3": ("R9 allows one edit too many",
            "if len(edits) > bound:", "if len(edits) > bound + 1:"),
    "PM4": ("R7 compares raw floats instead of micro-units",
            "if abs(q - before) > max_delta:",
            "if abs(float(parts[k][2]) - before / 1e6) > limits.max_abs_delta:"),
}


def _load_guard() -> Any:
    mutant = os.environ.get("VERIFIED_PY_MUTANT")
    if not mutant:
        return wsp_guard
    _, original, replacement = PY_MUTANTS[mutant]
    source = Path(wsp_guard.__file__).read_text(encoding="utf-8")
    assert source.count(original) == 1, f"mutant {mutant} no longer applies to wsp_guard.py"
    name = f"wsp_guard_mutant_{mutant}"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source.replace(original, replacement), name, "exec"), module.__dict__)  # noqa: S102
    return module


guard = _load_guard()

# ─── The kernel CLI ───────────────────────────────────────────────────────────────────────────


def _unavailable(reason: str) -> None:
    if REQUIRE:
        pytest.fail(f"REQUIRE_VERIFIED=1 but {reason}", pytrace=False)
    pytest.skip(reason)


def _kernel_binary() -> Path:
    env = os.environ.get("VERIFIED_KERNEL_BIN")
    if env:
        path = Path(env)
        if not path.is_file():
            _unavailable(f"VERIFIED_KERNEL_BIN={env} does not exist")
        return path
    cargo = shutil.which("cargo")
    if cargo is None:
        _unavailable("cargo is not installed, so the verified kernel can't be built")
    build = subprocess.run(
        [cargo, "build", "--release", "--locked", "--bin", "verified-kernel"],
        cwd=CRATE, capture_output=True, text=True, check=False,
        env={**os.environ, "CARGO_TARGET_DIR": str(CRATE / "target")},
    )
    if build.returncode != 0:
        _unavailable(f"cargo build of verified/ failed:\n{build.stderr[-2000:]}")
    exe = "verified-kernel.exe" if os.name == "nt" else "verified-kernel"
    return CRATE / "target" / "release" / exe


class Kernel:
    def __init__(self, path: Path) -> None:
        self.proc = subprocess.Popen(
            [str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("verified-kernel exited")
        reply = json.loads(line)
        assert "error" not in reply, f"kernel refused {request['op']}: {reply['error']}"
        return reply

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)


@pytest.fixture(scope="module")
def kernel():
    k = Kernel(_kernel_binary())
    yield k
    k.close()


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


NAN_BITS = bits(float("nan"))
MICRO = wsp_guard.MICRO

# What was exercised, for test_every_rule_was_exercised and the receipt.
SEEN: Counter[str] = Counter()

DIFF_SETTINGS = settings(
    max_examples=EXAMPLES, deadline=None, database=None, derandomize=REQUIRE,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large,
                           HealthCheck.filter_too_much],
)

# ─── Quantization ─────────────────────────────────────────────────────────────────────────────


@DIFF_SETTINGS
@given(st.one_of(
    st.floats(allow_nan=True, allow_infinity=True),
    # Above 2**52 a float is an integer and a + 0.5 is not representable: where a naive
    # floor(a + 0.5) double-rounds.
    st.integers(2**52, 2**53).map(lambda k: k / 1e6),
))
@example(0.5e-6)
@example(-0.5e-6)
@example(2.5e-6)
@example(4503599627.370497)
@example(-4503599627.370497)
@example(10.0000005)
@example(10.0000004)
@example(5e-324)
@example(1e300)
@example(-1e300)
@example(4e12)
@example(4.0000000000000005e12)
@example(-4.0000000000000005e12)
@example(9.007199254740993e9)
def test_quantize_agrees(kernel, x):
    assert guard.quantize(x) == kernel.call({"op": "quantize", "x": bits(x)})["q"]
    SEEN["quantize"] += 1


# ─── 1. The escalation gate ───────────────────────────────────────────────────────────────────

# Micro-unit amounts small enough that m / 1e6 round-trips exactly and float error stays far
# below the gap between a ratio and the gate's thresholds.
MICRO_LIMIT = 10**12
THRESHOLDS = (Fraction(3, 25), Fraction(1, 10), Fraction(1, 4))


@st.composite
def resource(draw) -> tuple[int, int]:
    """A (total, spent) pair in micro-units, biased toward the gate's thresholds."""
    total = draw(st.one_of(
        st.integers(1, MICRO_LIMIT), st.sampled_from([0, 1, 25, 100, 1_000_000]),
        st.integers(-1_000_000, 0),
    ))
    mode = draw(st.sampled_from(["any", "tie", "exhausted", "fresh"]))
    if mode == "tie" and total > 0:
        t = draw(st.sampled_from(THRESHOLDS))
        spent = total - (total * t.numerator) // t.denominator + draw(st.integers(-2, 2))
    elif mode == "exhausted":
        spent = total + draw(st.integers(0, 10**9))
    elif mode == "fresh":
        spent = 0
    else:
        spent = draw(st.integers(-1_000_000, MICRO_LIMIT))
    return total, max(-MICRO_LIMIT, min(MICRO_LIMIT, spent))


def exact_ratio(total: int, spent: int) -> Fraction:
    rem = max(total - spent, 0)
    if total <= 0:
        return Fraction(1 if rem > 0 else 0)
    return min(Fraction(rem, total), Fraction(1))


def _off_grid(draw) -> tuple[float, float]:
    """A (total, spent) pair of arbitrary floats, often a hair either side of a threshold."""
    total = draw(st.one_of(
        st.floats(-10.0, 1e7, allow_nan=False, allow_infinity=False),
        st.sampled_from([1.0, 0.3, 1.2, 90_000.0, 1e-7]),
    ))
    if draw(st.booleans()):
        t = draw(st.sampled_from(THRESHOLDS))
        jitter = draw(st.floats(-2e-6, 2e-6, allow_nan=False))
        spent = total - total * float(t) + jitter
    else:
        spent = draw(st.floats(-10.0, 1.1e7, allow_nan=False, allow_infinity=False))
    return total, spent


@st.composite
def gate_resource(draw) -> tuple[float, float]:
    """A (total, spent) pair as the engine holds it: on the micro-unit grid, or off it."""
    if draw(st.booleans()):
        total, spent = draw(resource())
        return total / 1e6, spent / 1e6
    return _off_grid(draw)


@st.composite
def gate_case(draw) -> dict[str, Any]:
    budget, latency = draw(gate_resource()), draw(gate_resource())
    return {
        "has_escalated": draw(st.booleans()),
        "allow_escalation": draw(st.booleans()),
        "solo_steps": draw(st.integers(0, 12)),
        "min_solo_steps": draw(st.integers(0, 4)),
        "step": draw(st.integers(0, 50)),
        "min_steps_before_terminate": draw(st.integers(0, 10)),
        "unresolved_subtasks": draw(st.sampled_from([0, 0, 1, 2, 5])),
        "budget": budget,
        "latency": latency,
    }


def python_legal(case: dict[str, Any]) -> list[str]:
    engine = object.__new__(OrchestrationEngine)
    engine.config = RunConfig(
        task="differential",
        allow_escalation=case["allow_escalation"],
        min_solo_steps=case["min_solo_steps"],
        min_steps_before_terminate=case["min_steps_before_terminate"],
    )
    (bt, bs), (lt, ls) = case["budget"], case["latency"]
    engine.state = OrchestratorState(
        step=case["step"],
        has_escalated=case["has_escalated"],
        solo_steps=case["solo_steps"],
        unresolved_subtasks=case["unresolved_subtasks"],
        budget_total_usd=bt, budget_spent_usd=bs,
        latency_budget_ms=lt, latency_consumed_ms=ls,
    )
    return [a.value for a in engine.legal_actions()]


def _quantized_ratio(pair: tuple[float, float]) -> Fraction:
    total, spent = pair
    return exact_ratio(guard.quantize(total), guard.quantize(spent))


def exact_legal(case: dict[str, Any]) -> list[str]:
    """legal_actions over exact rationals of the micro-unit amounts: the proved specification."""
    b, lat = _quantized_ratio(case["budget"]), _quantized_ratio(case["latency"])
    may_terminate = case["unresolved_subtasks"] == 0 or b < Fraction(1, 4) or lat < Fraction(1, 4)
    terminate = case["step"] >= case["min_steps_before_terminate"] and may_terminate
    if not case["has_escalated"]:
        legal = {Action.INVOKE_GENERALIST}
        if case["allow_escalation"] and case["solo_steps"] >= case["min_solo_steps"]:
            legal.add(Action.ESCALATE)
    else:
        legal = {Action.INVOKE_PLANNER, Action.INVOKE_RESEARCHER, Action.INVOKE_CRITIC,
                 Action.INVOKE_VERIFIER, Action.INVOKE_EXECUTOR, Action.INVOKE_MEMORY}
        if b > Fraction(3, 25) and lat > Fraction(1, 10):
            legal.add(Action.RUN_PARALLEL)
    if terminate:
        legal.add(Action.TERMINATE)
    return [a.value for a in ACTIONS if a in legal]


def at_tie(case: dict[str, Any]) -> bool:
    b, lat = _quantized_ratio(case["budget"]), _quantized_ratio(case["latency"])
    return b in (Fraction(3, 25), Fraction(1, 4)) or lat in (Fraction(1, 10), Fraction(1, 4))


def _on_grid(case: dict[str, Any]) -> bool:
    return all(guard.quantize(x) / 1e6 == x for pair in (case["budget"], case["latency"])
               for x in pair)


@DIFF_SETTINGS
@given(gate_case())
@example({"has_escalated": True, "allow_escalation": True, "solo_steps": 0, "min_solo_steps": 0,
          "step": 0, "min_steps_before_terminate": 0, "unresolved_subtasks": 1,
          # 0.12 left as a micro-unit amount, 0.1200004 as a float: the kernel says not above.
          "budget": (1.0, 0.8799996), "latency": (1.0, 0.0)})
@example({"has_escalated": True, "allow_escalation": True, "solo_steps": 0, "min_solo_steps": 0,
          "step": 0, "min_steps_before_terminate": 0, "unresolved_subtasks": 1,
          "budget": (1.0, 0.0), "latency": (1.0, 0.7500004)})
def test_gate_agrees(kernel, case):
    (bt, bs), (lt, ls) = case["budget"], case["latency"]
    request = {
        "op": "gate",
        **{k: case[k] for k in ("has_escalated", "allow_escalation", "solo_steps",
                                 "min_solo_steps", "step", "min_steps_before_terminate",
                                 "unresolved_subtasks")},
        "budget_total": bits(bt), "budget_spent": bits(bs),
        "latency_budget": bits(lt), "latency_consumed": bits(ls),
    }
    rust = kernel.call(request)["legal"]
    python = python_legal(case)
    # The kernel decides the thresholds exactly on micro-unit amounts, and so does the Python:
    # both quantise first, so even an exact tie (12% left against "more than 12%") agrees.
    assert rust == exact_legal(case)
    assert python == rust
    if at_tie(case):
        SEEN["gate_ties"] += 1
    if not _on_grid(case):
        SEEN["gate_off_grid"] += 1
    SEEN["gate_escalated" if case["has_escalated"] else "gate_solo"] += 1


# ─── 2. Budget accounting ─────────────────────────────────────────────────────────────────────


@DIFF_SETTINGS
@given(resource())
def test_remaining_and_normalized_agree(kernel, pair):
    total, spent = pair
    rust = kernel.call({"op": "budget", "total": bits(total / 1e6), "spent": bits(spent / 1e6)})
    state = OrchestratorState(
        budget_total_usd=total / 1e6, budget_spent_usd=spent / 1e6,
        latency_budget_ms=total / 1e6, latency_consumed_ms=spent / 1e6,
    )
    assert guard.quantize(state.budget_remaining_usd) == rust["remaining"]
    exact = exact_ratio(total, spent)
    assert rust["normalized"] == math.floor(exact * MICRO)
    for python in (state.budget_remaining, state.latency_remaining):
        # The kernel's value is the exact fraction rounded down to a micro-unit; the Python
        # float is the same fraction to within float error.
        assert 0.0 <= python <= 1.0
        assert rust["normalized"] - 1e-6 <= python * MICRO < rust["normalized"] + 1 + 1e-6
    SEEN["budget"] += 1


def _state(total: int, spent: int, lat_total: int, lat_spent: int) -> OrchestratorState:
    state = initial_state(task_complexity=0.5, budget_usd=total / 1e6,
                          latency_budget_ms=lat_total / 1e6, belief_dim=4,
                          rng=np.random.default_rng(0))
    state.budget_spent_usd = spent / 1e6
    state.latency_consumed_ms = lat_spent / 1e6
    return state


# Config and transition construction reject negative escalation costs (finding F-3).
@DIFF_SETTINGS
@given(resource(), resource(), st.integers(0, 10**9), st.integers(0, 10**9))
def test_escalation_spend_agrees(kernel, budget, latency, cost, lat_cost):
    state = _state(*budget, *latency)
    model = TransitionModel(belief_dim=4, escalation_cost_usd=cost / 1e6,
                            escalation_latency_ms=lat_cost / 1e6)
    nxt = model.sample(state, Action.ESCALATE, np.random.default_rng(1)).next_state
    for python, before, amount in ((nxt.budget_spent_usd, budget[1], cost),
                                   (nxt.latency_consumed_ms, latency[1], lat_cost)):
        rust = kernel.call({"op": "spend", "spent": bits(before / 1e6),
                            "cost": bits(amount / 1e6)})["spent"]
        assert guard.quantize(python) == rust
        assert rust >= before
    SEEN["spend_escalation"] += 1


@DIFF_SETTINGS
@given(resource(), resource(), st.integers(-10**9, 10**9), st.integers(1, 10**9))
def test_live_report_spend_agrees(kernel, budget, latency, cost, lat):
    state = _state(*budget, *latency)
    payload = {"outcome": "partial", "confidence": 0.5, "response": "r", "tokens": 10,
               "cost_usd": cost / 1e6, "latency_ms": lat / 1e6}
    if cost < 0:
        with pytest.raises(ValueError, match="cost_usd"):
            report_from_response(AGENTS["generalist"], state, payload, belief_dim=4)
        rust = kernel.call({"op": "spend", "spent": bits(budget[1] / 1e6),
                            "cost": bits(cost / 1e6)})["spent"]
        assert rust == budget[1]
        SEEN["spend_live_negative_cost"] += 1
        return
    report = report_from_response(AGENTS["generalist"], state, payload, belief_dim=4)
    nxt = TransitionModel(belief_dim=4).sample(
        state, Action.INVOKE_GENERALIST, np.random.default_rng(2),
        agent_ids=["generalist"], reports_override=[report],
    ).next_state
    for python, before, amount in ((nxt.budget_spent_usd, budget[1], cost),
                                   (nxt.latency_consumed_ms, latency[1], lat)):
        rust = kernel.call({"op": "spend", "spent": bits(before / 1e6),
                            "cost": bits(amount / 1e6)})["spent"]
        assert guard.quantize(python) == rust
    SEEN["spend_live"] += 1


# ─── 3. The WSP weight-patch guard ────────────────────────────────────────────────────────────

TENSORS = wsp_guard.TENSORS


I64_MIN, I64_MAX = -(2**63), 2**63 - 1


def _index_entry(i: Any) -> int | None:
    # The guard takes any Integral but bool (numpy.int64 included), so the adapter must too.
    # An integer past i64 cannot be in bounds for any shape; clamping it keeps it out of bounds
    # (R3 on both sides) instead of making the kernel's JSON parser refuse the request.
    if isinstance(i, bool) or not isinstance(i, numbers.Integral):
        return None
    return max(I64_MIN, min(I64_MAX, int(i)))


def _index_json(index: Any) -> Any:
    if not isinstance(index, (list, tuple)):
        return None
    return [_index_entry(i) for i in index]


def _value_bits(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return NAN_BITS
    try:
        return bits(float(value))
    except OverflowError:
        return NAN_BITS


def guard_request(case: dict[str, Any]) -> dict[str, Any]:
    """The adapter: the Python call's arguments, as the kernel CLI reads them."""
    limits = case["limits"]
    edits = []
    for e in case["edits"]:
        tensor, index, value = e["tensor"], e["index"], e["value"]
        edits.append({
            "tensor": tensor if isinstance(tensor, str) else None,
            "index": _index_json(index),
            "value": _value_bits(value),
        })
    return {
        "op": "guard",
        "shapes": {n: list(case["shapes"][n]) for n in TENSORS},
        "current": {n: [bits(x) for x in np.asarray(case["current"][n], dtype=float).ravel()]
                    for n in TENSORS},
        "edits": edits,
        "limits": {k: bits(getattr(limits, k))
                   for k in ("max_abs_value", "max_abs_delta", "max_changed_fraction")},
    }


def assert_guard_agrees(kernel: Kernel, case: dict[str, Any]) -> str:
    rust = kernel.call(guard_request(case))
    violation = guard.first_violation(case["shapes"], case["current"], case["edits"],
                                      case["limits"])
    verdict = guard.check_patch(case["shapes"], case["current"], case["edits"], case["limits"])
    python_rule = None if violation is None else violation.rule
    assert python_rule == rust["rule"], (violation, rust)
    assert verdict.rule == python_rule and verdict.accepted == (python_rule is None)
    if python_rule in ("R2", "R3", "R4", "R5", "R6", "R7", "R8"):
        assert violation.edit == rust["edit"]
    if python_rule is None:
        applied = guard.apply_patch(case["shapes"], case["current"], case["edits"],
                                    case["limits"])
        for name in TENSORS:
            ours = [guard.quantize(float(x)) for x in applied[name].ravel()]
            assert ours == rust["applied"][name], name
            assert applied[name].shape == tuple(case["shapes"][name])
    outcome = python_rule or "accepted"
    SEEN[outcome] += 1
    return outcome


def make_case(n: int, f: int, edits: list[dict[str, Any]], *, current: dict | None = None,
              limits: Any = None, fill: float = 0.5) -> dict[str, Any]:
    shapes = {"q_weights": (n, f), "synergy": (n, n), "terminate_weights": (f,)}
    cur = current or {name: np.full(shape, fill) for name, shape in shapes.items()}
    return {"shapes": shapes, "current": cur, "edits": edits,
            "limits": limits or wsp_guard.GuardLimits()}


def E(tensor: Any, index: Any, value: Any) -> dict[str, Any]:
    return {"tensor": tensor, "index": index, "value": value}


ALL = wsp_guard.GuardLimits(max_changed_fraction=1.0)
NAN_CURRENT = {"q_weights": np.full((3, 4), 0.5), "synergy": np.full((3, 3), 0.5),
               "terminate_weights": np.array([0.5, float("nan"), 0.5, 0.5])}

# (name, case, the rule the Python guard must give) — the edges the brief calls out.
EDGE_CASES = [
    ("empty", make_case(3, 4, []), "R1"),
    ("unknown tensor", make_case(3, 4, [E("bias", [0], 1.0)], limits=ALL), "R2"),
    ("tensor not a string", make_case(3, 4, [E(None, [0], 1.0)], limits=ALL), "R2"),
    ("R2 before R3", make_case(3, 4, [E("q_weights", [9, 9], 1.0), E("x", [0], 1.0)],
                               limits=ALL), "R2"),
    ("arity", make_case(3, 4, [E("q_weights", [1], 1.0)], limits=ALL), "R3"),
    ("row out of bounds", make_case(3, 4, [E("q_weights", [3, 0], 1.0)], limits=ALL), "R3"),
    ("negative index", make_case(3, 4, [E("terminate_weights", [-1], 1.0)], limits=ALL), "R3"),
    ("bool index", make_case(3, 4, [E("terminate_weights", [True], 1.0)], limits=ALL), "R3"),
    ("float index", make_case(3, 4, [E("terminate_weights", [1.0], 1.0)], limits=ALL), "R3"),
    ("2**70 index", make_case(3, 4, [E("terminate_weights", [2**70], 1.0)], limits=ALL), "R3"),
    ("-2**70 index", make_case(3, 4, [E("terminate_weights", [-(2**70)], 1.0)], limits=ALL),
     "R3"),
    ("2**64 index", make_case(3, 4, [E("q_weights", [0, 2**64], 1.0)], limits=ALL), "R3"),
    ("numpy int64 index", make_case(3, 4, [E("q_weights", [np.int64(1), np.int64(2)], 1.0)],
                                    limits=ALL), None),
    ("numpy int64 out of bounds", make_case(3, 4, [E("q_weights", [np.int64(3), 0], 1.0)],
                                            limits=ALL), "R3"),
    ("index not a list", make_case(3, 4, [E("terminate_weights", 1, 1.0)], limits=ALL), "R3"),
    ("synergy diagonal", make_case(3, 4, [E("synergy", [1, 1], 1.0)], limits=ALL), "R4"),
    ("synergy lower triangle", make_case(3, 4, [E("synergy", [2, 0], 1.0)], limits=ALL), "R4"),
    ("synergy upper", make_case(3, 4, [E("synergy", [0, 2], 1.0)], limits=ALL), None),
    ("NaN", make_case(3, 4, [E("q_weights", [0, 0], float("nan"))], limits=ALL), "R5"),
    ("inf", make_case(3, 4, [E("q_weights", [0, 0], float("inf"))], limits=ALL), "R5"),
    ("-inf", make_case(3, 4, [E("q_weights", [0, 0], float("-inf"))], limits=ALL), "R5"),
    ("string value", make_case(3, 4, [E("q_weights", [0, 0], "1.0")], limits=ALL), "R5"),
    ("bool value", make_case(3, 4, [E("q_weights", [0, 0], True)], limits=ALL), "R5"),
    ("int past float", make_case(3, 4, [E("q_weights", [0, 0], 10**400)], limits=ALL), "R5"),
    ("1e300", make_case(3, 4, [E("q_weights", [0, 0], 1e300)], limits=ALL), "R6"),
    ("-1e300", make_case(3, 4, [E("q_weights", [0, 0], -1e300)], limits=ALL), "R6"),
    ("|value| == max", make_case(3, 4, [E("q_weights", [0, 0], -10.0)], fill=-8.0,
                                 limits=ALL), None),
    ("|value| just over max", make_case(3, 4, [E("q_weights", [0, 0], 10.0000005)], fill=8.0,
                                        limits=ALL), "R6"),
    ("|value| over by < half a micro", make_case(3, 4, [E("q_weights", [0, 0], 10.0000004)],
                                                fill=8.0, limits=ALL), None),
    ("delta == max", make_case(3, 4, [E("q_weights", [0, 0], 5.5)], limits=ALL), None),
    ("delta just over max", make_case(3, 4, [E("q_weights", [0, 0], 5.5000005)], limits=ALL),
     "R7"),
    ("replaces NaN", make_case(3, 4, [E("terminate_weights", [1], 0.5)], current=NAN_CURRENT,
                               limits=ALL), "R7"),
    ("duplicate", make_case(3, 4, [E("q_weights", [0, 0], 1.0), E("q_weights", (0, 0), 2.0)],
                            limits=ALL), "R8"),
    ("same index, other tensor", make_case(3, 4, [E("q_weights", [0, 1], 1.0),
                                                 E("synergy", [0, 1], 1.0)], limits=ALL), None),
    ("R9 at the bound", make_case(3, 4, [E("terminate_weights", [0], 1.0)]), None),
    ("R9 one over", make_case(3, 4, [E("terminate_weights", [0], 1.0),
                                     E("terminate_weights", [1], 1.0)]), "R9"),
    ("R9 bound of zero", make_case(1, 1, [E("terminate_weights", [0], 1.0)]), "R9"),
]


@pytest.mark.parametrize(("name", "case", "expected"), EDGE_CASES,
                         ids=[c[0] for c in EDGE_CASES])
def test_guard_edge_cases(kernel, name, case, expected):
    assert assert_guard_agrees(kernel, case) == (expected or "accepted")


def _micro_float(draw, lo: float, hi: float) -> float:
    return draw(st.integers(int(lo * MICRO), int(hi * MICRO))) / MICRO


@st.composite
def guard_case(draw) -> dict[str, Any]:
    n = draw(st.integers(0, 6))
    f = draw(st.integers(0, 10))
    shapes = {"q_weights": (n, f), "synergy": (n, n), "terminate_weights": (f,)}
    limits = wsp_guard.GuardLimits(
        max_abs_value=draw(st.sampled_from([10.0, 1.0, 0.0])) if draw(st.booleans())
        else _micro_float(draw, 0, 20),
        max_abs_delta=draw(st.sampled_from([5.0, 0.5, 0.0])) if draw(st.booleans())
        else _micro_float(draw, 0, 20),
        max_changed_fraction=draw(st.sampled_from([0.05, 0.1, 0.25, 1.0, 0.0])),
    )

    def current_value() -> float:
        roll = draw(st.integers(0, 40))
        if roll == 0:
            return draw(st.sampled_from([float("nan"), float("inf"), float("-inf")]))
        if roll == 1:
            return draw(st.floats(-12, 12))
        return _micro_float(draw, -12, 12)

    current = {name: np.array([current_value() for _ in range(math.prod(shape))],
                              dtype=float).reshape(shape)
               for name, shape in shapes.items()}

    def valid_index(tensor: str) -> list[int] | None:
        dims = shapes[tensor]
        if any(d == 0 for d in dims) or (tensor == "synergy" and dims[0] < 2):
            return None
        if tensor == "synergy":
            i = draw(st.integers(0, dims[0] - 2))
            return [i, draw(st.integers(i + 1, dims[0] - 1))]
        return [draw(st.integers(0, d - 1)) for d in dims]

    edits: list[dict[str, Any]] = []
    count = draw(st.integers(1, 6)) if draw(st.integers(0, 15)) else 0
    for _ in range(count):
        tensor = draw(st.sampled_from(TENSORS))
        index = valid_index(tensor)
        if index is None:
            if f == 0:
                break
            tensor, index = "terminate_weights", [draw(st.integers(0, f - 1))]
        before = float(np.asarray(current[tensor])[tuple(index)])
        value: Any = before if math.isfinite(before) else 0.0
        value += _micro_float(draw, -limits.max_abs_delta, limits.max_abs_delta)
        edits.append(E(tensor, index, value))

    # Then break at most a couple of things, so most patches get deep into the rules.
    for _ in range(draw(st.integers(0, 2))):
        if not edits:
            break
        k = draw(st.integers(0, len(edits) - 1))
        e = dict(edits[k])
        kind = draw(st.sampled_from([
            "tensor", "arity", "bounds", "negative", "bool", "huge", "diagonal", "lower",
            "nan", "inf", "big", "abs-edge", "delta-edge", "half-micro", "duplicate",
        ]))
        if kind == "tensor":
            e["tensor"] = draw(st.sampled_from(["bias", "Q_WEIGHTS", "", None, 3]))
        elif kind == "arity":
            e["index"] = list(e["index"]) + [0] if draw(st.booleans()) else list(e["index"])[:-1]
        elif kind == "bounds":
            e["index"] = [d for d in shapes.get(e["tensor"], (1,))]
        elif kind == "negative":
            e["index"] = [-1] + list(e["index"])[1:]
        elif kind == "bool":
            e["index"] = [True] + list(e["index"])[1:]
        elif kind == "huge":
            e["index"] = [2**70] + list(e["index"])[1:]
        elif kind in ("diagonal", "lower") and n >= 1:
            i = draw(st.integers(0, n - 1))
            e["tensor"], e["index"] = "synergy", [i, draw(st.integers(0, i))] if kind == "lower" \
                else [i, i]
        elif kind == "nan":
            e["value"] = float("nan")
        elif kind == "inf":
            e["value"] = draw(st.sampled_from([float("inf"), float("-inf")]))
        elif kind == "big":
            e["value"] = draw(st.sampled_from([1e300, -1e300, 4e12, 2.0**63, 10**30]))
        elif kind == "abs-edge":
            e["value"] = draw(st.sampled_from([1, -1])) * limits.max_abs_value + draw(
                st.sampled_from([0.0, 1e-6, -1e-6, 4e-7, 5e-7]))
        elif kind == "delta-edge" and isinstance(e["index"], list):
            try:
                before = float(np.asarray(current[e["tensor"]])[tuple(e["index"])])
            except (IndexError, KeyError, TypeError):
                before = 0.0
            e["value"] = before + draw(st.sampled_from([1, -1])) * limits.max_abs_delta + draw(
                st.sampled_from([0.0, 1e-6, -1e-6, 4e-7, 5e-7]))
        elif kind == "half-micro":
            e["value"] = float(e["value"]) + 5e-7 if isinstance(e["value"], float) else 0.5
        elif kind == "duplicate" and len(edits) > 1:
            other = edits[draw(st.integers(0, len(edits) - 1))]
            e["tensor"], e["index"] = other["tensor"], list(other["index"])
        edits[k] = e
    return {"shapes": shapes, "current": current, "edits": edits, "limits": limits}


@DIFF_SETTINGS
@given(guard_case())
def test_guard_agrees(kernel, case):
    assert_guard_agrees(kernel, case)


# ─── 3b. The production apply path (wsp/apply.py), not just the guard module ─────────────────


@st.composite
def production_patch(draw) -> list[Any]:
    """Edits against the real policy's shapes, mostly near-valid so many are accepted."""
    from app.orchestration.wsp import Edit

    rows, cols = len(AGENTS), FEATURE_DIM
    edits = []
    for _ in range(draw(st.integers(1, 9))):
        tensor = draw(st.sampled_from(TENSORS))
        if tensor == "q_weights":
            index = (draw(st.integers(0, rows)), draw(st.integers(0, cols)))
        elif tensor == "synergy":
            i, j = draw(st.integers(0, rows - 1)), draw(st.integers(0, rows - 1))
            index = (min(i, j), max(i, j)) if draw(st.integers(0, 5)) else (i, j)
        else:
            index = (draw(st.integers(0, cols)),)
        value = draw(st.one_of(
            st.floats(-12, 12, allow_nan=False),
            st.integers(-12 * MICRO, 12 * MICRO).map(lambda m: m / MICRO),
        ))
        edits.append(Edit(tensor=tensor, index=index, value=value))
    return edits


@DIFF_SETTINGS
@given(production_patch(), st.sampled_from([0.05, 0.1, 1.0]))
def test_production_apply_agrees(kernel, edits, fraction):
    """``wsp.apply.apply_patch`` — what run creation calls — against the kernel's guard+apply."""
    from app.orchestration.policies import create_policy
    from app.orchestration.wsp import Patch
    from app.orchestration.wsp.apply import PatchRejected, apply_patch, tensors_of

    policy = create_policy("markov_game", feature_dim=FEATURE_DIM)
    before = tensors_of(policy)
    limits = wsp_guard.GuardLimits(max_changed_fraction=fraction)
    case = {"shapes": {n: tuple(t.shape) for n, t in before.items()},
            "current": before, "edits": [e.to_dict() for e in edits], "limits": limits}
    rust = kernel.call(guard_request(case))
    patch = Patch(goal="differential", rationale="r", edits=tuple(edits))
    try:
        applied = apply_patch(policy, patch, limits)
    except PatchRejected as exc:
        assert rust["rule"] == exc.verdict.rule, (exc.verdict, rust)
        SEEN["production_rejected"] += 1
        return
    assert rust["rule"] is None
    after = tensors_of(applied.policy)
    for name in TENSORS:
        assert [guard.quantize(float(x)) for x in after[name].ravel()] == rust["applied"][name]
    # The policy it was given is untouched; the copy carries the patch.
    for name in TENSORS:
        np.testing.assert_array_equal(tensors_of(policy)[name], before[name])
    SEEN["production_accepted"] += 1


def test_every_rule_was_exercised():
    """The guard tests above reached every rule, and acceptance, at least once."""
    if not SEEN:
        pytest.skip("run with the rest of this module")
    missing = [r for r in (*wsp_guard.RULES, "accepted", "gate_off_grid", "gate_ties",
                           "production_accepted", "production_rejected") if SEEN[r] == 0]
    assert not missing, f"never exercised: {missing}; seen {dict(SEEN)}"
    out = os.environ.get("VERIFIED_RESULTS")
    if out:
        Path(out).write_text(json.dumps({"examples": EXAMPLES, "seen": dict(sorted(SEEN.items()))},
                                        indent=2), encoding="utf-8")
