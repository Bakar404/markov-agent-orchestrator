"""Weight-space programming from the bridge: ask a Copilot model to write the patch.

The arena renders the base policy's weights and calibration diagnostics (``GET /api/wsp/brief``,
calibration seeds only); a ``GitHubCopilotAgent`` reads them and proposes a patch toward the
goal; the patch goes back with every ``wsp_markov_game`` run, where the arena's guard vets it.
Programming happens once per experiment, not per seed, so every seed evaluates the same network.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .arena import Arena

_BACKEND = Path(__file__).resolve().parents[2] / "backend"

MAX_SEED = 2**31 - 1
CALIBRATION_SEED_BAND = range(2_000_000_000, 2_000_001_000)
"""Mirrors ``app.orchestration.wsp.calibrate.CALIBRATION_SEED_BAND``; a backend test pins the two
together. Copied rather than imported because that module needs numpy and the engine."""


def check_evaluation_seeds(seeds: list[int]) -> None:
    """Refuse a seed the arena would refuse, before anything is spent on an editor call."""
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed {seed!r} must be an integer in [0, {MAX_SEED}]")
        if seed in CALIBRATION_SEED_BAND:
            raise ValueError(
                f"seed {seed} is reserved for weight-space-programming calibration "
                f"({CALIBRATION_SEED_BAND.start}..{CALIBRATION_SEED_BAND.stop - 1})"
            )


def _editor_module():
    # The patch contract and the editor live in the backend so there is exactly one copy of
    # each; they import only the standard library, so the bridge venv can load them.
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))
    from app.orchestration.wsp import editor

    return editor


async def program_patch(
    arena: Arena, *, model: str, goal: str | None = None, max_steps: int = 40,
    belief_dim: int = 8, budget: float | None = None, latency_budget_ms: float | None = None,
) -> dict:
    """Ask the editor for a patch and return the run fields every programmed seed sends.

    The brief is calibrated under the arm's own budget and latency budget, so the diagnostics
    the editor reads describe the episodes the arm will actually be evaluated on. The result
    carries the goal the patch was written for (the arena checks the patch against it) and the
    provenance the arena records with the run: the base checkpoint's hash and the calibration
    seeds the editor saw.
    """
    params: dict = {"max_steps": max_steps, "belief_dim": belief_dim}
    if goal:
        params["goal"] = goal
    if budget is not None:
        params["budget_usd"] = budget
    if latency_budget_ms is not None:
        params["latency_budget_ms"] = latency_budget_ms
    brief = arena.wsp_brief(**params)
    editor = _editor_module().CopilotEditor(model=model)
    patch = await editor.apropose(brief["rendered"], brief["goal"])
    return {
        "wsp_patch": patch.to_dict(),
        "wsp_goal": brief["goal"],
        "wsp_provenance": {
            "base_sha256": brief["base"]["sha256"],
            "calibration_seeds": list(brief["calibration"]["seeds"]),
        },
    }


def programmed_run_fields(wsp: dict | None) -> dict:
    """The fields a programmed arm adds to ``POST /api/runs``: all three, or none.

    Sending the patch without its goal makes the arena replay it against the strategy's default
    goal and refuse it; sending it without provenance is refused outright.
    """
    if not wsp:
        return {}
    return {key: wsp[key] for key in ("wsp_patch", "wsp_goal", "wsp_provenance")}
