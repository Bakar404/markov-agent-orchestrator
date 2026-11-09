"""The weight patch contract.

A patch is exactly::

    {"goal": str, "rationale": str,
     "edits": [{"tensor": "q_weights"|"synergy"|"terminate_weights",
                "index": [int, ...], "value": float}]}

``value`` is the *new absolute value* of the entry, not a delta. Parsing is strict: unknown or
missing keys, an unknown tensor, a non-integer index or a non-finite value are format errors.
Whether the edit is *allowed* (bounds, shape, how much of the network it touches) is the
guard's decision, not this module's.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

TENSORS: tuple[str, ...] = ("q_weights", "synergy", "terminate_weights")

_PATCH_KEYS = frozenset({"goal", "rationale", "edits"})
_EDIT_KEYS = frozenset({"tensor", "index", "value"})


class PatchFormatError(ValueError):
    """The payload does not match the patch contract."""


@dataclass(frozen=True)
class Edit:
    tensor: str
    index: tuple[int, ...]
    value: float

    def to_dict(self) -> dict[str, Any]:
        return {"tensor": self.tensor, "index": list(self.index), "value": float(self.value)}

    @classmethod
    def from_dict(cls, data: Any) -> Edit:
        if not isinstance(data, dict):
            raise PatchFormatError("each edit must be an object")
        keys = set(data)
        if keys != _EDIT_KEYS:
            missing = sorted(_EDIT_KEYS - keys)
            extra = sorted(keys - _EDIT_KEYS)
            raise PatchFormatError(f"edit keys must be exactly {sorted(_EDIT_KEYS)}; "
                                   f"missing={missing} extra={extra}")
        tensor = data["tensor"]
        if tensor not in TENSORS:
            raise PatchFormatError(f"unknown tensor {tensor!r}; expected one of {list(TENSORS)}")
        index = data["index"]
        if not isinstance(index, list) or not index:
            raise PatchFormatError("index must be a non-empty list of integers")
        if any(isinstance(i, bool) or not isinstance(i, int) for i in index):
            raise PatchFormatError("index entries must be integers")
        value = data["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PatchFormatError("value must be a number")
        try:
            number = float(value)
        except OverflowError as exc:
            # An int past float range (10**400) is as non-finite as infinity itself.
            raise PatchFormatError("value must be finite") from exc
        if not math.isfinite(number):
            raise PatchFormatError("value must be finite")
        return cls(tensor=tensor, index=tuple(index), value=number)


@dataclass(frozen=True)
class Patch:
    goal: str
    rationale: str
    edits: tuple[Edit, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "rationale": self.rationale,
            "edits": [edit.to_dict() for edit in self.edits],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Any) -> Patch:
        if not isinstance(data, dict):
            raise PatchFormatError("patch must be a JSON object")
        keys = set(data)
        if keys != _PATCH_KEYS:
            missing = sorted(_PATCH_KEYS - keys)
            extra = sorted(keys - _PATCH_KEYS)
            raise PatchFormatError(f"patch keys must be exactly {sorted(_PATCH_KEYS)}; "
                                   f"missing={missing} extra={extra}")
        goal, rationale, edits = data["goal"], data["rationale"], data["edits"]
        if not isinstance(goal, str) or not isinstance(rationale, str):
            raise PatchFormatError("goal and rationale must be strings")
        if not isinstance(edits, list):
            raise PatchFormatError("edits must be a list")
        return cls(goal=goal, rationale=rationale,
                   edits=tuple(Edit.from_dict(edit) for edit in edits))

    @classmethod
    def from_json(cls, text: str) -> Patch:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PatchFormatError(f"patch is not valid JSON: {exc}") from exc
        return cls.from_dict(data)
