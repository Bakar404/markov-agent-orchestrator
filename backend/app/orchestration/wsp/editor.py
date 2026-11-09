"""Weight editors: whatever turns (rendered weights, goal) into a :class:`Patch`.

* :class:`RecordedEditor` replays a patch saved to disk. It is what CI and offline runs use, and
  it keeps a programmed arm reproducible: the same recorded patch yields the same network.
* :class:`CopilotEditor` asks a general LLM through Agent Framework's ``GitHubCopilotAgent``.
  ``agent_framework`` is imported lazily, so the backend does not depend on it; only the
  bridge's virtualenv needs it. Prompt construction and reply parsing are pure functions so they
  can be tested offline.

This module avoids numpy and the engine so the bridge can import it without the backend stack.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .patch import Patch, PatchFormatError

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
_EXAMPLE_NAME = re.compile(r"^[a-z0-9_]{1,64}$")


@runtime_checkable
class WeightEditor(Protocol):
    name: str

    def propose(self, rendered: str, goal: str) -> Patch: ...


def _normalise(text: str) -> str:
    return " ".join(text.split())


def example_path(name: str) -> Path:
    """Resolve a shipped example patch by name. Names only — never a caller-supplied path."""
    if not _EXAMPLE_NAME.match(name):
        raise PatchFormatError(f"invalid example patch name {name!r}")
    path = EXAMPLES_DIR / f"{name}.json"
    if not path.is_file():
        raise PatchFormatError(f"no example patch named {name!r}")
    return path


class RecordedEditor:
    """Replays a recorded patch. Refuses one recorded for a different goal."""

    name = "recorded"

    def __init__(self, source: str | Path | dict[str, Any] | Patch) -> None:
        if isinstance(source, Patch):
            self.patch = source
        elif isinstance(source, dict):
            self.patch = Patch.from_dict(source)
        else:
            self.patch = Patch.from_json(Path(source).read_text(encoding="utf-8"))

    @classmethod
    def example(cls, name: str) -> RecordedEditor:
        return cls(example_path(name))

    def propose(self, rendered: str, goal: str) -> Patch:
        if _normalise(self.patch.goal) != _normalise(goal):
            raise PatchFormatError(
                f"recorded patch was written for goal {self.patch.goal!r}, not {goal!r}"
            )
        return self.patch


EDITOR_INSTRUCTIONS = """\
You are programming a small neural policy by editing its weights directly. There is no \
gradient descent: you read the weights, reason about how they are used, and choose new values.

Rules:
- Reply with ONE JSON object and nothing else: no prose, no markdown fences.
- The object has exactly these keys:
  {"goal": <the goal, verbatim>, "rationale": <why these edits achieve it>,
   "edits": [{"tensor": "q_weights"|"synergy"|"terminate_weights",
              "index": [int, ...], "value": <new absolute value, a number>}]}
- "value" is the entry's NEW value, not a delta.
- q_weights takes [row, column]; synergy takes [i, j] with i < j; terminate_weights takes [column].
- Respect every edit limit stated in the weights. A guard rejects the whole patch otherwise.
- Prefer the fewest edits that achieve the goal.
"""


def build_prompt(rendered: str, goal: str) -> str:
    return (
        f"GOAL\n{goal}\n\n"
        f"WEIGHTS\n{rendered}\n\n"
        "Return the JSON patch now. Copy the goal verbatim into the \"goal\" field."
    )


MAX_REPLY_CHARS = 64_000
"""A patch that fits the guard's limits is a few kilobytes; anything far larger is not one."""

_FENCED = re.compile(r"\A```(?:json)?[ \t]*\n(.*)\n[ \t]*```\Z", re.DOTALL)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise PatchFormatError("reply repeats a key; refusing to guess which one was meant")
    return dict(pairs)


def _no_constants(name: str) -> Any:
    raise PatchFormatError(f"reply contains {name}, which is not JSON")


def parse_reply(text: str) -> Patch:
    """Parse an LLM reply as a patch, strictly.

    The reply must be one JSON object and nothing else; the only allowance is a single code
    fence wrapping the whole reply, because models add one even when told not to. Prose around
    the object, a second object, a repeated key, ``NaN``/``Infinity`` or a reply over
    :data:`MAX_REPLY_CHARS` is refused rather than searched for something patch-shaped. Parsing
    is ``json.loads`` only, and :meth:`Patch.from_dict` admits exactly goal, rationale and
    edits, so nothing the model writes can reach limits, seeds or file paths: those come from the
    caller and the guard.
    """
    if not isinstance(text, str):
        raise PatchFormatError("reply must be text")
    if len(text) > MAX_REPLY_CHARS:
        raise PatchFormatError(f"reply is over {MAX_REPLY_CHARS} characters")
    body = text.strip()
    fenced = _FENCED.match(body)
    if fenced:
        body = fenced.group(1).strip()
    if not body.startswith("{"):
        raise PatchFormatError("reply must be a single JSON object and nothing else")
    try:
        data = json.loads(
            body, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constants
        )
    except json.JSONDecodeError as exc:
        raise PatchFormatError(f"reply is not a single JSON object: {exc.msg}") from exc
    except RecursionError as exc:
        raise PatchFormatError("reply is nested too deeply") from exc
    return Patch.from_dict(data)


class CopilotEditor:
    """A general LLM, reached through ``agent_framework.github.GitHubCopilotAgent``."""

    name = "copilot"

    def __init__(self, model: str = "", timeout: float = 600.0) -> None:
        try:
            from agent_framework.github import GitHubCopilotAgent, GitHubCopilotOptions
        except ImportError as exc:  # pragma: no cover - depends on the bridge venv
            raise RuntimeError(
                "CopilotEditor needs agent_framework (install the bridge requirements)"
            ) from exc
        options = GitHubCopilotOptions(model=model, timeout=timeout) if model else \
            GitHubCopilotOptions(timeout=timeout)
        self._agent = GitHubCopilotAgent(
            name="weight_editor",
            instructions=EDITOR_INSTRUCTIONS,
            default_options=options,
        )
        self.model = model

    async def apropose(self, rendered: str, goal: str) -> Patch:
        response = await self._agent.run(build_prompt(rendered, goal))
        patch = parse_reply(response.text or "")
        if _normalise(patch.goal) != _normalise(goal):
            raise PatchFormatError("editor rewrote the goal; refusing its patch")
        return patch

    def propose(self, rendered: str, goal: str) -> Patch:
        return asyncio.run(self.apropose(rendered, goal))
