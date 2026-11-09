"""Apply a weight patch — through the guard, to a copy, with WSP-style edit metrics.

Order matters: :func:`apply_patch` asks :func:`app.orchestration.wsp_guard.check_patch` for a
verdict *before* touching anything, and refuses the patch unless the guard accepts it. A guard
that is missing or not yet implemented counts as a rejection (fail closed). The caller's policy
is never mutated; the patched network is a deep copy.

Metrics follow the paper's reporting: how many entries changed, what fraction of effective parameters
that is, the relative Frobenius norm of the change per tensor, and the largest individual edits.
The headline counts only parameters the policy reads (synergy's upper triangle among the agents a
coalition can field); raw metrics retain the full stored tensors.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import wsp_guard
from ..agents import AGENT_IDS, SOLO_AGENT
from ..state import FEATURE_NAMES
from .patch import TENSORS, Edit, Patch

try:
    from ..wsp_guard import GuardLimits
except ImportError:  # pragma: no cover - the guard module may define only check_patch

    @dataclass(frozen=True)
    class GuardLimits:  # type: ignore[no-redef]
        max_abs_value: float = 10.0
        max_abs_delta: float = 5.0
        max_changed_fraction: float = 0.05


@dataclass(frozen=True)
class Verdict:
    """Local mirror of the guard's verdict, used when the guard could not produce one."""

    accepted: bool
    reason: str | None = None
    rule: str | None = None


def verdict_dict(verdict: Any) -> dict[str, Any]:
    return {
        "accepted": bool(getattr(verdict, "accepted", False)),
        "reason": getattr(verdict, "reason", None),
        "rule": getattr(verdict, "rule", None),
    }


def limits_dict(limits: Any) -> dict[str, float]:
    return {
        "max_abs_value": float(limits.max_abs_value),
        "max_abs_delta": float(limits.max_abs_delta),
        "max_changed_fraction": float(limits.max_changed_fraction),
    }


class PatchRejected(ValueError):
    def __init__(self, verdict: Any) -> None:
        self.verdict = verdict
        info = verdict_dict(verdict)
        super().__init__(f"patch rejected ({info['rule']}): {info['reason']}")


@dataclass
class AppliedPatch:
    policy: Any
    patch: Patch
    verdict: Any
    metrics: dict[str, Any]
    limits: Any = field(default_factory=GuardLimits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch": self.patch.to_dict(),
            "verdict": verdict_dict(self.verdict),
            "metrics": self.metrics,
            "limits": limits_dict(self.limits),
        }


def tensors_of(policy: Any) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(policy, name), dtype=float) for name in TENSORS}


def install_weights(policy: Any, state: dict[str, Any]) -> None:
    """Copy the programmable weight tensors of ``state`` into ``policy``, and nothing else.

    ``state`` is a programmed policy state (the output of :func:`apply_patch`, already vetted by
    the guard) or a base checkpoint. Hyperparameters — temperature, learning rate, discount —
    are left exactly as the run's ``policy_options`` built them; loading a whole state dict
    used to reset them silently to whatever the base checkpoint carried.
    """
    staged: dict[str, np.ndarray] = {}
    for name in TENSORS:
        current = np.asarray(getattr(policy, name), dtype=float)
        if name not in state:
            raise ValueError(f"programmed state has no {name}")
        tensor = np.array(state[name], dtype=float)
        if tensor.shape != current.shape:
            raise ValueError(
                f"programmed {name} has shape {list(tensor.shape)}, expected {list(current.shape)}"
            )
        if not np.isfinite(tensor).all():
            raise ValueError(f"programmed {name} is not finite")
        staged[name] = tensor
    for name, tensor in staged.items():
        setattr(policy, name, tensor)


def entry_label(tensor: str, index: tuple[int, ...]) -> str:
    def name(axis_names: tuple[str, ...] | list[str], i: int) -> str:
        return axis_names[i] if 0 <= i < len(axis_names) else str(i)

    if tensor == "q_weights" and len(index) == 2:
        return f"q_weights[{name(AGENT_IDS, index[0])}, {name(FEATURE_NAMES, index[1])}]"
    if tensor == "synergy" and len(index) == 2:
        return f"synergy[{name(AGENT_IDS, index[0])}, {name(AGENT_IDS, index[1])}]"
    if tensor == "terminate_weights" and len(index) == 1:
        return f"terminate_weights[{name(FEATURE_NAMES, index[0])}]"
    return f"{tensor}{list(index)}"


def _structural_problem(edits: tuple[Edit, ...], shapes: dict[str, tuple[int, ...]]) -> Verdict:
    """Defence in depth after the guard: anything that would not land on a live entry."""
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for edit in edits:
        shape = shapes.get(edit.tensor)
        if shape is None or len(edit.index) != len(shape):
            return Verdict(False, f"{edit.tensor}{list(edit.index)} has the wrong rank", "shape")
        if any(not 0 <= i < n for i, n in zip(edit.index, shape, strict=True)):
            return Verdict(False, f"{edit.tensor}{list(edit.index)} is out of bounds", "shape")
        if edit.tensor == "synergy" and edit.index[0] >= edit.index[1]:
            return Verdict(
                False, f"synergy{list(edit.index)} is never read; use [i, j] with i < j",
                "synergy_upper_triangle",
            )
        key = (edit.tensor, edit.index)
        if key in seen:
            return Verdict(False, f"{edit.tensor}{list(edit.index)} edited twice", "duplicate")
        seen.add(key)
    return Verdict(True)


def _effective_mask(name: str, tensor: np.ndarray) -> np.ndarray:
    """Which stored entries the policy can actually read.

    Synergy is read only at [i, j] with i < j, and only for coalitions, which never include the
    generalist — so its row and column are dead weight. Every q and terminate weight is read.
    """
    if name != "synergy" or tensor.ndim != 2:
        return np.ones(tensor.shape, dtype=bool)
    mask = np.triu(np.ones(tensor.shape, dtype=bool), k=1)
    if tensor.shape == (len(AGENT_IDS), len(AGENT_IDS)):
        solo = AGENT_IDS.index(SOLO_AGENT)
        mask[solo, :] = False
        mask[:, solo] = False
    return mask


def edit_metrics(
    before: dict[str, np.ndarray], after: dict[str, np.ndarray], top: int = 10
) -> dict[str, Any]:
    total = sum(int(t.size) for t in before.values())
    # The headline fraction is over parameters the policy reads: every q/terminate weight and
    # the synergy pairs a coalition can field (upper triangle, generalist excluded). The guard's
    # R9 limit deliberately keeps the raw stored total as its denominator, exactly as proved in
    # verified/, so `changed_fraction_raw` is the figure that limit is checked against.
    effective_total = sum(int(_effective_mask(name, t).sum()) for name, t in before.items())
    per_tensor: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    changed_total = 0
    for name in TENSORS:
        b, a = before[name], after[name]
        delta = a - b
        mask = delta != 0
        changed = int(mask.sum())
        changed_total += changed
        dw = float(np.linalg.norm(delta))
        base = float(np.linalg.norm(b))
        per_tensor[name] = {
            "shape": list(b.shape),
            "parameters": int(b.size),
            "changed": changed,
            "changed_pct": 100.0 * changed / b.size if b.size else 0.0,
            "dw_fro": dw,
            "w_fro": base,
            "relative_dw_fro": dw / base if base > 0 else None,
        }
        for idx in zip(*np.nonzero(mask), strict=True):
            index = tuple(int(i) for i in idx)
            entries.append({
                "tensor": name,
                "index": list(index),
                "label": entry_label(name, index),
                "before": float(b[index]),
                "after": float(a[index]),
                "delta": float(delta[index]),
            })
    entries.sort(key=lambda e: -abs(e["delta"]))
    all_delta = math.sqrt(sum(t["dw_fro"] ** 2 for t in per_tensor.values()))
    all_base = math.sqrt(sum(t["w_fro"] ** 2 for t in per_tensor.values()))
    return {
        "total_parameters": total,
        "effective_parameters": effective_total,
        "changed_entries": changed_total,
        "changed_fraction": changed_total / effective_total if effective_total else 0.0,
        "changed_fraction_raw": changed_total / total if total else 0.0,
        "changed_pct": 100.0 * changed_total / effective_total if effective_total else 0.0,
        "dw_fro": all_delta,
        "relative_dw_fro": all_delta / all_base if all_base > 0 else None,
        "per_tensor": per_tensor,
        "top_changed": entries[:top],
    }


def apply_patch(policy: Any, patch: Patch, limits: Any | None = None) -> AppliedPatch:
    """Vet ``patch`` with the guard, then apply it to a copy of ``policy``.

    Raises :class:`PatchRejected` (carrying the verdict) if the guard rejects the patch, if the
    guard is unavailable, or if an accepted edit would not land on a live entry.
    """
    limits = limits if limits is not None else GuardLimits()
    before = tensors_of(policy)
    shapes = {name: tuple(t.shape) for name, t in before.items()}
    current = {name: t.copy() for name, t in before.items()}
    try:
        verdict = wsp_guard.check_patch(shapes, current, list(patch.edits), limits)
    except NotImplementedError:
        verdict = Verdict(False, "the weight guard is not implemented yet", "guard_unavailable")
    if not getattr(verdict, "accepted", False):
        raise PatchRejected(verdict)

    structural = _structural_problem(patch.edits, shapes)
    if not structural.accepted:
        raise PatchRejected(structural)

    patched = copy.deepcopy(policy)
    # The write itself is the guard's apply — the function the Rust kernel's apply is proved
    # and differentially tested against — so production never has a second, unproved copy.
    try:
        after = wsp_guard.apply_patch(shapes, current, list(patch.edits), limits)
    except wsp_guard.GuardRejected as exc:
        raise PatchRejected(exc.verdict) from exc
    for name, tensor in after.items():
        setattr(patched, name, tensor)
    return AppliedPatch(
        policy=patched,
        patch=patch,
        verdict=verdict,
        metrics=edit_metrics(before, after),
        limits=limits,
    )
