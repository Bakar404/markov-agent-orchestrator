"""The WSP weight-patch guard: the gate an LLM's weight edit must pass before it touches the policy.

A patch is a list of edits, each ``{"tensor": name, "index": [int, ...], "value": float}``, aimed
at the three tensors the ``markov_game`` policy reads: ``q_weights`` (agents x features),
``synergy`` (agents x agents, only ``i < j`` is ever read) and ``terminate_weights`` (features).
:func:`check_patch` checks nine rules, in order, and the first one broken rejects the patch:

====  =========================================================================================
R1    the patch has at least one edit
R2    every edit names one of the three tensors
R3    every index has one integer per dimension of its tensor, each within bounds
R4    every synergy edit is strictly above the diagonal (``i < j``); the policy never reads the
      diagonal or the lower triangle, so editing them would change nothing but the change count
R5    every value is a finite number
R6    ``|value| <= max_abs_value``
R7    the entry being replaced is finite, and ``|value - current| <= max_abs_delta``
R8    no two edits target the same ``(tensor, index)``
R9    ``len(edits) <= floor(max_changed_fraction x total entries of the three tensors)``
====  =========================================================================================

Rules are checked rule by rule, not edit by edit: a patch with an R6 problem in its first edit
and an R3 problem in its last is rejected under R3.

Values, limits and the fraction are compared in fixed-point micro-units, ``round(x * 1e6)``
rounded half away from zero, never as raw floats. That is what lets this module agree bit for bit
with the Verus-proved Rust kernel in ``verified/``, which states these rules as a specification
and proves its implementation meets it; ``backend/tests/test_verified_kernel.py`` checks that
the two agree. A consequence worth knowing: values closer than half a micro-unit to a limit are
judged after rounding (``10.0000004`` passes a limit of ``10``; ``10.0000005`` does not).

Malformed *edits* are verdicts, never exceptions: that is the guard's job. Malformed *inputs* from
the caller (missing tensors, wrong ranks, current values that don't match the shapes, invalid
limits) are programming errors and raise :class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

TENSORS: tuple[str, ...] = ("q_weights", "synergy", "terminate_weights")
RANKS: dict[str, int] = {"q_weights": 2, "synergy": 2, "terminate_weights": 1}
RULES: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9")

MICRO = 1_000_000
"""One whole unit, in micro-units."""

MICRO_MAX = 4_000_000_000_000_000_000
"""The largest magnitude a quantized value takes; larger values saturate here."""

MAX_DIM = 1 << 20
"""No tensor dimension may exceed this (the Rust kernel's proofs assume it)."""


@dataclass(frozen=True)
class Edit:
    """One weight edit. ``value`` is the new absolute value of the entry, not a delta."""

    tensor: str
    index: tuple[int, ...]
    value: float


@dataclass(frozen=True)
class GuardLimits:
    max_abs_value: float = 10.0
    max_abs_delta: float = 5.0
    max_changed_fraction: float = 0.05

    def __post_init__(self) -> None:
        _validate_limits(self)


@dataclass(frozen=True)
class GuardVerdict:
    accepted: bool
    reason: str | None = None
    rule: str | None = None


@dataclass(frozen=True)
class Violation:
    """The first rule a patch breaks. ``edit`` is the offending edit's position for R2 to R8
    (for R8, the later of the two duplicates) and ``None`` for R1 and R9."""

    rule: str
    edit: int | None
    reason: str


class GuardRejected(ValueError):
    """Raised by :func:`apply_patch` when the guard rejects the patch."""

    def __init__(self, verdict: GuardVerdict) -> None:
        super().__init__(verdict.reason)
        self.verdict = verdict


def quantize(x: Any) -> int | None:
    """``round(x * 1e6)``, half away from zero, saturating at +-MICRO_MAX; ``None`` if ``x`` is
    not a finite number. The Rust kernel's ``quantize`` is the same function."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    try:
        x = float(x)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    y = x * 1e6
    if not math.isfinite(y):
        return MICRO_MAX if y > 0 else -MICRO_MAX
    a = abs(y)
    f = math.floor(a)
    r = f + 1 if a - f >= 0.5 else f
    r = r if y >= 0 else -r
    return max(-MICRO_MAX, min(MICRO_MAX, r))


def check_patch(
    shapes: Mapping[str, tuple[int, ...]],
    current: Mapping[str, Any],
    edits: list[Any],
    limits: GuardLimits | None = None,
) -> GuardVerdict:
    """Vet a weight patch against R1 to R9. Accepted only if every rule holds."""
    violation = first_violation(shapes, current, edits, limits)
    if violation is None:
        return GuardVerdict(True, None, None)
    return GuardVerdict(False, violation.reason, violation.rule)


def first_violation(
    shapes: Mapping[str, tuple[int, ...]],
    current: Mapping[str, Any],
    edits: list[Any],
    limits: GuardLimits | None = None,
) -> Violation | None:
    """The first rule the patch breaks, and which edit breaks it; ``None`` if it passes."""
    limits = limits if limits is not None else GuardLimits()
    edits = list(edits)
    if not edits:
        return Violation("R1", None, "R1: the patch has no edits")

    _validate_limits(limits)
    dims = _validate_shapes(shapes)
    cur = _validate_current(current, dims)
    max_abs = quantize(limits.max_abs_value)
    max_delta = quantize(limits.max_abs_delta)
    fraction = quantize(limits.max_changed_fraction)
    assert max_abs is not None and max_delta is not None and fraction is not None

    parts = [_parts(e) for e in edits]

    for k, (tensor, _, _) in enumerate(parts):
        if not _known(tensor):
            return Violation("R2", k, f"R2: edit {k} names unknown tensor {tensor!r}; "
                                      f"expected one of {list(TENSORS)}")

    indices: list[tuple[int, ...]] = []
    for k, (tensor, index, _) in enumerate(parts):
        ix = _index(index, dims[tensor])
        if ix is None:
            return Violation("R3", k, f"R3: edit {k} index {index!r} does not fit {tensor} "
                                      f"of shape {dims[tensor]}")
        indices.append(ix)

    for k, (tensor, _, _) in enumerate(parts):
        if tensor == "synergy" and not indices[k][0] < indices[k][1]:
            return Violation("R4", k, f"R4: edit {k} synergy{list(indices[k])} is on or below "
                                      "the diagonal, which the policy never reads; use i < j")

    values: list[int] = []
    for k, (_, _, value) in enumerate(parts):
        q = quantize(value)
        if q is None:
            return Violation("R5", k, f"R5: edit {k} value {value!r} is not a finite number")
        values.append(q)

    for k, q in enumerate(values):
        if abs(q) > max_abs:
            return Violation("R6", k, f"R6: edit {k} |value| {parts[k][2]!r} exceeds "
                                      f"max_abs_value {limits.max_abs_value!r}")

    for k, q in enumerate(values):
        tensor = parts[k][0]
        before = cur[tensor][indices[k]]
        if before is None:
            return Violation("R7", k, f"R7: edit {k} replaces a non-finite entry "
                                      f"{tensor}{list(indices[k])}")
        if abs(q - before) > max_delta:
            return Violation("R7", k, f"R7: edit {k} moves {tensor}{list(indices[k])} by more "
                                      f"than max_abs_delta {limits.max_abs_delta!r}")

    seen: set[tuple[str, tuple[int, ...]]] = set()
    for k, (tensor, _, _) in enumerate(parts):
        key = (tensor, indices[k])
        if key in seen:
            return Violation("R8", k, f"R8: edit {k} repeats {tensor}{list(indices[k])}")
        seen.add(key)

    total = sum(math.prod(dims[name]) for name in TENSORS)
    bound = (fraction * total) // MICRO
    if len(edits) > bound:
        return Violation("R9", None, f"R9: {len(edits)} edits exceed the bound of {bound} "
                                     f"(max_changed_fraction {limits.max_changed_fraction!r} "
                                     f"of {total} entries)")
    return None


def apply_patch(
    shapes: Mapping[str, tuple[int, ...]],
    current: Mapping[str, Any],
    edits: list[Any],
    limits: GuardLimits | None = None,
) -> dict[str, np.ndarray]:
    """Apply a patch the guard accepts to copies of the three tensors; raise
    :class:`GuardRejected` if it doesn't. ``current`` is never modified."""
    verdict = check_patch(shapes, current, edits, limits)
    if not verdict.accepted:
        raise GuardRejected(verdict)
    out = {name: np.array(current[name], dtype=float, copy=True) for name in TENSORS}
    for edit in edits:
        tensor, index, value = _parts(edit)
        out[tensor][tuple(int(i) for i in index)] = float(value)
    return out


# ─── Internals ─────────────────────────────────────────────────────────────────────────────


def _parts(edit: Any) -> tuple[Any, Any, Any]:
    if isinstance(edit, Mapping):
        return edit.get("tensor"), edit.get("index"), edit.get("value")
    return (getattr(edit, "tensor", None), getattr(edit, "index", None),
            getattr(edit, "value", None))


def _known(tensor: Any) -> bool:
    return isinstance(tensor, str) and tensor in TENSORS


def _index(index: Any, dims: tuple[int, ...]) -> tuple[int, ...] | None:
    if not isinstance(index, (list, tuple)) or len(index) != len(dims):
        return None
    out: list[int] = []
    for i, n in zip(index, dims, strict=True):
        if isinstance(i, bool) or not isinstance(i, numbers.Integral):
            return None
        i = int(i)
        if not 0 <= i < n:
            return None
        out.append(i)
    return tuple(out)


def _validate_limits(limits: Any) -> None:
    for name in ("max_abs_value", "max_abs_delta", "max_changed_fraction"):
        value = getattr(limits, name, None)
        if quantize(value) is None or float(value) < 0:
            raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    if float(limits.max_changed_fraction) > 1.0:
        raise ValueError("max_changed_fraction must be at most 1")


def _validate_shapes(shapes: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    dims: dict[str, tuple[int, ...]] = {}
    for name in TENSORS:
        if name not in shapes:
            raise ValueError(f"shapes is missing {name}")
        shape = tuple(shapes[name])
        if len(shape) != RANKS[name]:
            raise ValueError(f"{name} must have rank {RANKS[name]}, got shape {shape}")
        if any(isinstance(d, bool) or not isinstance(d, numbers.Integral) for d in shape):
            raise ValueError(f"{name} shape must be integers, got {shape}")
        shape = tuple(int(d) for d in shape)
        if any(not 0 <= d <= MAX_DIM for d in shape):
            raise ValueError(f"{name} dimensions must be between 0 and {MAX_DIM}, got {shape}")
        dims[name] = shape
    if dims["synergy"][0] != dims["synergy"][1]:
        raise ValueError(f"synergy must be square, got {dims['synergy']}")
    return dims


def _validate_current(
    current: Mapping[str, Any], dims: dict[str, tuple[int, ...]]
) -> dict[str, np.ndarray]:
    """The current weights, quantized (``None`` marks a non-finite entry)."""
    out: dict[str, np.ndarray] = {}
    for name in TENSORS:
        if name not in current:
            raise ValueError(f"current is missing {name}")
        try:
            arr = np.asarray(current[name], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"current {name} is not numeric: {exc}") from exc
        if arr.shape != dims[name]:
            raise ValueError(f"current {name} has shape {arr.shape}, expected {dims[name]}")
        quantized = np.empty(arr.shape, dtype=object)
        for ix, x in np.ndenumerate(arr):
            quantized[ix] = quantize(float(x))
        out[name] = quantized
    return out
