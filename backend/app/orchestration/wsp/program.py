"""End-to-end weight-space programming of the markov_game policy.

    base checkpoint ─► calibrate (calibration seeds) ─► render ─► editor.propose(goal)
                   ─► apply_patch (guard first) ─► programmed policy state

The base is a trained markov_game checkpoint when one exists, otherwise the policy's default
initialisation. The checkpoint is read from ``$WSP_CHECKPOINT`` or
``<data>/checkpoints/markov_game.json`` — a server-side path only; HTTP callers can never name
a file. Either a policy ``state_dict`` or a full engine snapshot (with ``policy_state``) works.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...config import DATA_DIR
from ..state import FEATURE_DIM
from .apply import AppliedPatch, GuardLimits, apply_patch, limits_dict
from .calibrate import DEFAULT_CALIBRATION_SEEDS, SeedSplit, calibrate
from .editor import RecordedEditor, WeightEditor
from .patch import TENSORS, Patch
from .render import render_policy

DEFAULT_GOAL = (
    "Escalate only when the generalist has stalled and the remaining hypotheses justify the "
    "coalition's cost; prefer the narrowest coalition that resolves them"
)
DEFAULT_EXAMPLE = "narrow_coalitions"
POLICY_ID = "markov_game"


class CheckpointError(ValueError):
    pass


_CALIBRATION_CACHE: dict[tuple[str, tuple[int, ...], str], dict[str, Any]] = {}
_CALIBRATION_CACHE_SIZE = 64


def checkpoint_path() -> Path:
    override = os.getenv("WSP_CHECKPOINT")
    return Path(override) if override else DATA_DIR / "checkpoints" / f"{POLICY_ID}.json"


def _fresh_policy() -> Any:
    from ..policies import create_policy

    return create_policy(POLICY_ID, feature_dim=FEATURE_DIM)


def _tensor_digest(policy: Any) -> str:
    """SHA-256 of the programmable tensors, for a base that has no checkpoint file to hash."""
    canonical = json.dumps(
        {name: np.asarray(getattr(policy, name), dtype=float).tolist() for name in TENSORS},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_base() -> tuple[Any, dict[str, Any]]:
    """The policy to program, and a record of where it came from.

    The record carries a SHA-256 — of the checkpoint file's bytes, or of the default tensors —
    so a stored patch says exactly which weights it was written against. Error messages name
    only the file, never its server-side directory.
    """
    policy = _fresh_policy()
    path = checkpoint_path()
    if not path.is_file():
        return policy, {"source": "default_init", "checkpoint": None,
                        "sha256": _tensor_digest(policy)}
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CheckpointError(
            f"cannot read checkpoint {path.name}: {type(exc).__name__}"
        ) from exc
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CheckpointError(f"checkpoint {path.name} is not valid JSON") from exc
    state = payload.get("policy_state", payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise CheckpointError("checkpoint is not a policy state or engine snapshot")
    if state.get("id", POLICY_ID) != POLICY_ID:
        raise CheckpointError("checkpoint is not a markov_game policy")
    expected = {name: np.asarray(getattr(policy, name)).shape for name in TENSORS}
    for name, shape in expected.items():
        if name in state and np.asarray(state[name]).shape != shape:
            raise CheckpointError(f"checkpoint {name} has shape "
                                  f"{list(np.asarray(state[name]).shape)}, expected {list(shape)}")
    try:
        policy.load_state_dict(state)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CheckpointError(f"checkpoint {path.name} has malformed policy state") from exc
    return policy, {"source": "checkpoint", "checkpoint": path.name,
                    "sha256": hashlib.sha256(raw).hexdigest()}


def make_editor(programming: dict[str, Any], patch: dict[str, Any] | None = None) -> WeightEditor:
    """Pick the editor for a run. An inline patch (e.g. from the Copilot bridge) wins."""
    if patch is not None:
        editor = RecordedEditor(patch)
        editor.name = "inline"
        return editor
    kind = programming.get("editor", "recorded")
    if kind == "recorded":
        return RecordedEditor.example(programming.get("patch", DEFAULT_EXAMPLE))
    if kind == "copilot":
        from .editor import CopilotEditor

        return CopilotEditor(model=programming.get("model", ""))
    raise ValueError(f"unknown weight editor {kind!r}")


@dataclass
class Brief:
    """Everything an editor sees: rendered weights plus the diagnostics behind them."""

    policy: Any
    base: dict[str, Any]
    goal: str
    calibration: dict[str, Any]
    rendered: str
    limits: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "base": self.base,
            "calibration": self.calibration,
            "rendered": self.rendered,
            "limits": limits_dict(self.limits),
        }


def _calibrate_cached(
    policy: Any, base: dict[str, Any], seeds: tuple[int, ...], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Calibration is deterministic in (base weights, seeds, task config), so it is memoised.

    Each call runs one sampled episode per seed synchronously; without the cache, every brief
    and every seed of a WSP arm would repeat the same eight episodes.
    """
    key = (base["sha256"], tuple(seeds), json.dumps(overrides, sort_keys=True, default=str))
    cached = _CALIBRATION_CACHE.get(key)
    if cached is None:
        cached = calibrate(policy.state_dict(), seeds, **overrides)
        if len(_CALIBRATION_CACHE) >= _CALIBRATION_CACHE_SIZE:
            _CALIBRATION_CACHE.pop(next(iter(_CALIBRATION_CACHE)))
        _CALIBRATION_CACHE[key] = copy.deepcopy(cached)
    return copy.deepcopy(cached)


def brief(
    goal: str = DEFAULT_GOAL,
    calibration_seeds: Iterable[int] = DEFAULT_CALIBRATION_SEEDS,
    evaluation_seeds: Iterable[int] = (),
    limits: Any | None = None,
    base_sha256: str | None = None,
    **overrides: Any,
) -> Brief:
    """Load the base, calibrate it under ``overrides`` and render it for an editor.

    ``base_sha256`` pins the base: if the checkpoint no longer hashes to it, nothing is
    programmed, because a patch of absolute values means something else on other weights.
    """
    split = SeedSplit.build(evaluation=evaluation_seeds, calibration=calibration_seeds)
    limits = limits if limits is not None else GuardLimits()
    policy, base = load_base()
    if base_sha256 is not None and base["sha256"] != base_sha256:
        raise CheckpointError(
            f"the base checkpoint changed: it hashes to {base['sha256'][:12]}…, but this patch "
            f"or experiment is pinned to {str(base_sha256)[:12]}…"
        )
    report = _calibrate_cached(policy, base, split.calibration, overrides)
    return Brief(
        policy=policy,
        base=base,
        goal=goal,
        calibration=report,
        rendered=render_policy(policy, report, limits),
        limits=limits,
    )


@dataclass
class Programmed:
    brief: Brief
    editor: str
    applied: AppliedPatch

    @property
    def policy_state(self) -> dict[str, Any]:
        return self.applied.policy.state_dict()


def program(
    editor: WeightEditor,
    goal: str = DEFAULT_GOAL,
    calibration_seeds: Iterable[int] = DEFAULT_CALIBRATION_SEEDS,
    evaluation_seeds: Iterable[int] = (),
    limits: Any | None = None,
    base_sha256: str | None = None,
    **overrides: Any,
) -> Programmed:
    """Calibrate, ask ``editor`` for a patch toward ``goal``, and apply it through the guard."""
    b = brief(goal, calibration_seeds, evaluation_seeds, limits, base_sha256, **overrides)
    patch: Patch = editor.propose(b.rendered, goal)
    applied = apply_patch(b.policy, patch, b.limits)
    return Programmed(brief=b, editor=getattr(editor, "name", type(editor).__name__),
                      applied=applied)
