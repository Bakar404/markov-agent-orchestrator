"""Calibration: what the base policy actually does, measured on calibration seeds only.

Seed split
----------
Weight-space programming is a form of fitting: the editor looks at behaviour and changes the
network. If it saw behaviour on the seeds the arm is later scored on, the edit could be tuned to
those episodes, and the paired comparison would no longer measure the arm against held-out
tasks. So the seeds are split:

* **Calibration seeds** live in a reserved band, :data:`CALIBRATION_SEED_BAND`
  (``2_000_000_000 .. 2_000_000_999``). Only these are run to produce the diagnostics the editor
  reads.
* **Evaluation seeds** are every other seed. The run service refuses to create or reset any run
  on a seed inside the band, and redraws random seeds that land in it.

:class:`SeedSplit` enforces both halves and that the two sets are disjoint.

The summary reports *behaviour* — how often the policy escalates, when, with how many agents,
why episodes end, what they cost. It deliberately omits internal reward: the arena never ranks
on it, and handing it to the editor would invite programming the policy toward the reward
model rather than toward the stated goal.
"""

from __future__ import annotations

import copy
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

CALIBRATION_SEED_BAND = range(2_000_000_000, 2_000_001_000)
DEFAULT_CALIBRATION_SEEDS: tuple[int, ...] = tuple(CALIBRATION_SEED_BAND[:8])

_OVERRIDE_KEYS = (
    "task_complexity",
    "max_steps",
    "belief_dim",
    "task_shape",
    "budget_usd",
    "latency_budget_ms",
    "stochasticity",
    "confidence_target",
    "verification_target",
    "min_steps_before_terminate",
    "policy_options",
    "escalation_cost_usd",
    "escalation_latency_ms",
    "min_solo_steps",
    "allow_escalation",
)
"""Every config field that shapes an episode. Calibration runs under the evaluation arm's own
settings — a budget or latency cap that ends episodes early changes what the editor should see —
and only the seed (reserved band), the mode (sampled) and the task text differ."""


class SeedLeakError(ValueError):
    """Calibration and evaluation seeds overlap, or a seed is on the wrong side of the band."""


def is_calibration_seed(seed: int) -> bool:
    return seed in CALIBRATION_SEED_BAND


@dataclass(frozen=True)
class SeedSplit:
    calibration: tuple[int, ...]
    evaluation: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.calibration:
            raise SeedLeakError("calibration needs at least one seed")
        stray = [s for s in self.calibration if not is_calibration_seed(s)]
        if stray:
            raise SeedLeakError(
                f"calibration seeds must lie in {CALIBRATION_SEED_BAND.start}.."
                f"{CALIBRATION_SEED_BAND.stop - 1}; got {stray}"
            )
        leaked = [s for s in self.evaluation if is_calibration_seed(s)]
        if leaked:
            raise SeedLeakError(f"evaluation seeds {leaked} are reserved for calibration")
        overlap = set(self.calibration) & set(self.evaluation)
        if overlap:
            raise SeedLeakError(f"seeds {sorted(overlap)} used for both calibration and evaluation")

    @classmethod
    def build(
        cls, evaluation: Iterable[int] = (), calibration: Iterable[int] | None = None
    ) -> SeedSplit:
        cal = tuple(calibration) if calibration is not None else DEFAULT_CALIBRATION_SEEDS
        return cls(calibration=cal, evaluation=tuple(evaluation))


def calibration_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """The parts of a run config that shape the episode, copied onto calibration episodes."""
    return {k: copy.deepcopy(config[k]) for k in _OVERRIDE_KEYS if k in config}


def _episode(policy_state: dict[str, Any], seed: int, overrides: dict[str, Any]) -> dict[str, Any]:
    from ..engine import OrchestrationEngine, RunConfig
    from .apply import install_weights

    config = RunConfig(task="wsp calibration", policy="markov_game", seed=seed, mode="sim",
                       **copy.deepcopy(overrides))
    engine = OrchestrationEngine(config)
    # Weights only: the evaluation run keeps its own policy_options, so calibration must too.
    install_weights(engine.policy, policy_state)
    escalated_at: int | None = None
    coalitions: list[tuple[str, ...]] = []
    steps = 0
    while not engine.done and steps < config.max_steps + 5:
        result = engine.step()
        steps += 1
        if result.action == "escalate" and escalated_at is None:
            escalated_at = result.step
        if len(result.agents) > 1:
            coalitions.append(tuple(sorted(result.agents)))
    return {
        "seed": seed,
        "steps": steps,
        "reason": engine.state.termination_reason or "none",
        "escalated_at": escalated_at,
        "coalitions": coalitions,
        "cost": engine.total_cost,
        "tokens": engine.total_tokens,
    }


def calibrate(
    policy_state: dict[str, Any],
    seeds: Iterable[int] = DEFAULT_CALIBRATION_SEEDS,
    **overrides: Any,
) -> dict[str, Any]:
    """Run the policy described by ``policy_state`` on calibration seeds and summarise it.

    ``policy_state`` is copied into a fresh engine per episode, so the caller's policy is never
    mutated by the online learning that happens during an episode.
    """
    split = SeedSplit.build(calibration=seeds)
    episodes = [_episode(policy_state, seed, overrides) for seed in split.calibration]
    n = len(episodes)
    escalated = [e for e in episodes if e["escalated_at"] is not None]
    sizes = Counter(len(c) for e in episodes for c in e["coalitions"])
    teams = Counter("+".join(c) for e in episodes for c in e["coalitions"])
    return {
        "seeds": list(split.calibration),
        "episodes": n,
        "escalation_rate": len(escalated) / n,
        "mean_steps_to_escalate": (
            statistics.mean(e["escalated_at"] for e in escalated) if escalated else None
        ),
        "termination_reasons": dict(Counter(e["reason"] for e in episodes).most_common()),
        "mean_steps": statistics.mean(e["steps"] for e in episodes),
        "coalition_sizes": {str(k): v for k, v in sorted(sizes.items())},
        "coalition_steps_per_episode": sum(sizes.values()) / n,
        "top_coalitions": [{"agents": k, "count": v} for k, v in teams.most_common(5)],
        "mean_cost_usd": statistics.mean(e["cost"] for e in episodes),
        "mean_tokens": statistics.mean(e["tokens"] for e in episodes),
        "overrides": copy.deepcopy(dict(overrides)),
    }
