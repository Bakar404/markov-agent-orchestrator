"""Cross-episode learning experiments.

Shared by the CLI (``tools/campaign.py``) and the API so the numbers in the UI and the numbers
in the terminal come from one implementation.

Each policy is run twice over the *same* sequence of task instances: once carrying learned
parameters between episodes, once with a fresh policy each time. Reporting the paired
difference removes task difficulty as a confound, which matters because per-episode reward is
high variance.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np

from ..orchestration.engine import OrchestrationEngine, RunConfig
from ..orchestration.policies import POLICY_REGISTRY, policy_catalog

MAX_EPISODES = 200


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    reward: float
    won: bool
    steps: int
    confidence: float
    cost: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "seed": self.seed,
            "reward": self.reward,
            "won": self.won,
            "steps": self.steps,
            "confidence": self.confidence,
            "cost": self.cost,
            "reason": self.reason,
        }


@dataclass
class ArmSummary:
    mean_reward: float
    win_rate: float
    mean_steps: float
    mean_confidence: float
    episodes: list[EpisodeResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mean_reward": self.mean_reward,
            "win_rate": self.win_rate,
            "mean_steps": self.mean_steps,
            "mean_confidence": self.mean_confidence,
            "episodes": [e.to_dict() for e in self.episodes],
        }


def _task_shape_for(seed: int, spread: float) -> dict[str, float]:
    """Deterministic task shape for an episode.

    Derived from the episode seed, so the carried and fresh arms see identical task instances
    and the paired difference stays exact. ``spread`` of 0 pins every episode to the neutral
    0.5, which is the behaviour the published campaign table was measured under.
    """
    if spread <= 0.0:
        return {}
    rng = np.random.default_rng(seed ^ 0x5EED)
    half = min(spread, 0.5)
    return {
        name: float(np.clip(rng.uniform(0.5 - half, 0.5 + half), 0.0, 1.0))
        for name in ("needs_evidence", "needs_execution", "needs_verification")
    }


def _run_episode(
    policy_id: str,
    seed: int,
    carried_state: dict | None,
    overrides: dict,
    shape_spread: float = 0.0,
) -> tuple[EpisodeResult, dict]:
    config = RunConfig(
        task="campaign",
        policy=policy_id,
        seed=seed,
        task_shape=_task_shape_for(seed, shape_spread),
        **overrides,
    )
    engine = OrchestrationEngine(config)
    if carried_state:
        engine.policy.load_state_dict(carried_state)

    while not engine.done:
        engine.step()

    state = engine.state
    result = EpisodeResult(
        episode=0,
        seed=seed,
        reward=engine.cumulative_reward,
        won=state.termination_reason == "goal_reached",
        steps=state.step,
        confidence=state.confidence,
        cost=engine.total_cost,
        reason=state.termination_reason or "none",
    )
    return result, engine.policy.state_dict()


def _run_arm(
    policy_id: str,
    episodes: int,
    seed_base: int,
    carry: bool,
    overrides: dict,
    shape_spread: float = 0.0,
) -> ArmSummary:
    results: list[EpisodeResult] = []
    carried: dict | None = None

    for index in range(episodes):
        result, policy_state = _run_episode(
            policy_id,
            seed_base + index,
            carried if carry else None,
            overrides,
            shape_spread,
        )
        result.episode = index
        results.append(result)
        carried = policy_state

    rewards = [r.reward for r in results]
    return ArmSummary(
        mean_reward=statistics.mean(rewards),
        win_rate=sum(r.won for r in results) / len(results),
        mean_steps=statistics.mean(r.steps for r in results),
        mean_confidence=statistics.mean(r.confidence for r in results),
        episodes=results,
    )


def _slope(rewards: list[float]) -> float:
    """Least-squares reward-per-episode trend."""
    if len(rewards) < 2:
        return 0.0
    x = np.arange(len(rewards), dtype=float)
    return float(np.polyfit(x, np.asarray(rewards, dtype=float), 1)[0])


def _block_means(rewards: list[float], blocks: int = 3) -> list[float]:
    size = max(len(rewards) // blocks, 1)
    out = []
    for i in range(blocks):
        chunk = rewards[i * size : (i + 1) * size]
        if chunk:
            out.append(statistics.mean(chunk))
    return out


def run_campaign(
    *,
    policies: list[str] | None = None,
    episodes: int = 40,
    seed_base: int = 1000,
    max_steps: int = 40,
    budget_usd: float = 1.2,
    task_complexity: float = 0.55,
    task_shape_spread: float = 0.0,
) -> dict:
    """Run the paired carried-vs-fresh experiment for each requested policy."""
    episodes = max(2, min(episodes, MAX_EPISODES))
    selected = policies or ["contextual_bandit", "mdp", "markov_game", "marl"]

    unknown = [p for p in selected if p not in POLICY_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown policies: {', '.join(unknown)}")

    overrides = {
        "max_steps": max_steps,
        "budget_usd": budget_usd,
        "task_complexity": task_complexity,
    }
    labels = {entry["id"]: entry for entry in policy_catalog()}

    results = []
    for policy_id in selected:
        carried = _run_arm(policy_id, episodes, seed_base, True, overrides, task_shape_spread)
        fresh = _run_arm(policy_id, episodes, seed_base, False, overrides, task_shape_spread)

        carried_rewards = [e.reward for e in carried.episodes]
        fresh_rewards = [e.reward for e in fresh.episodes]
        deltas = [c - f for c, f in zip(carried_rewards, fresh_rewards, strict=True)]

        mean_delta = statistics.mean(deltas)
        # Paired standard error: the episodes share a seed, so the pairing is exact.
        stderr = statistics.pstdev(deltas) / max(len(deltas) ** 0.5, 1e-9)

        results.append(
            {
                "policy": policy_id,
                "label": labels.get(policy_id, {}).get("label", policy_id),
                "stage": labels.get(policy_id, {}).get("stage", 0),
                "carried": carried.to_dict(),
                "fresh": fresh.to_dict(),
                "delta": mean_delta,
                "stderr": stderr,
                "significant": bool(abs(mean_delta) > 2 * stderr and stderr > 0),
                "slope": _slope(carried_rewards),
                "blocks": _block_means(carried_rewards),
            }
        )

    return {
        "config": {
            "episodes": episodes,
            "seed_base": seed_base,
            "max_steps": max_steps,
            "budget_usd": budget_usd,
            "task_complexity": task_complexity,
            "task_shape_spread": task_shape_spread,
            "policies": selected,
        },
        "results": results,
        "interpretation": (
            "Delta is carried minus fresh reward on identical task instances. "
            "A policy is flagged significant when the paired difference exceeds two standard "
            "errors. Baselines carry no state, so their delta must be exactly zero - that is "
            "the control."
        ),
    }
