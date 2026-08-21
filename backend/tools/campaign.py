"""Cross-episode learning harness.

The single-episode balance probe cannot answer the question the whole policy stack exists to
answer: *does carrying learned parameters across episodes make the orchestrator better?*

Each policy is run twice over the same sequence of task instances:

* ``carried``  - policy parameters persist from one episode to the next
* ``fresh``    - a new policy every episode (the control)

Seeds vary per episode, so improvement means generalization across task instances rather than
memorization of one. Reporting the paired difference removes the task difficulty as a
confound, which matters because episode rewards are high variance.

    python -m tools.campaign --episodes 40 --policies mdp,markov_game,marl
"""

from __future__ import annotations

import argparse
import statistics

import numpy as np

from app.orchestration.engine import OrchestrationEngine, RunConfig
from app.orchestration.policies import POLICY_REGISTRY


def run_episode(
    policy_id: str,
    seed: int,
    carried_state: dict | None,
    **overrides,
) -> tuple[dict, dict]:
    """Run one episode; return its outcome and the policy state to carry forward."""
    config = RunConfig(task="campaign", policy=policy_id, seed=seed, **overrides)
    engine = OrchestrationEngine(config)
    if carried_state:
        engine.policy.load_state_dict(carried_state)

    while not engine.done:
        engine.step()

    state = engine.state
    outcome = {
        "reward": engine.cumulative_reward,
        "won": state.termination_reason == "goal_reached",
        "steps": state.step,
        "confidence": state.confidence,
        "cost": engine.total_cost,
        "reason": state.termination_reason or "none",
    }
    return outcome, engine.policy.state_dict()


def run_campaign(policy_id: str, episodes: int, seed_base: int, carry: bool, **overrides) -> list[dict]:
    results: list[dict] = []
    carried: dict | None = None
    for index in range(episodes):
        outcome, policy_state = run_episode(
            policy_id, seed_base + index, carried if carry else None, **overrides
        )
        outcome["episode"] = index
        results.append(outcome)
        carried = policy_state
    return results


def slope(rewards: list[float]) -> float:
    """Least-squares reward-per-episode trend."""
    if len(rewards) < 2:
        return 0.0
    x = np.arange(len(rewards), dtype=float)
    y = np.asarray(rewards, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def block_means(rewards: list[float], blocks: int = 3) -> list[float]:
    size = max(len(rewards) // blocks, 1)
    return [
        statistics.mean(rewards[i * size : (i + 1) * size])
        for i in range(blocks)
        if rewards[i * size : (i + 1) * size]
    ]


def summarize(policy_id: str, carried: list[dict], fresh: list[dict]) -> str:
    carried_rewards = [r["reward"] for r in carried]
    fresh_rewards = [r["reward"] for r in fresh]

    # Paired: same seed, same task instance, so the difference isolates the learning effect.
    deltas = [c - f for c, f in zip(carried_rewards, fresh_rewards, strict=True)]
    mean_delta = statistics.mean(deltas)
    stderr = statistics.pstdev(deltas) / max(len(deltas) ** 0.5, 1e-9)

    blocks = block_means(carried_rewards)
    trend = " → ".join(f"{b:+.2f}" for b in blocks)

    carried_wins = sum(r["won"] for r in carried) / len(carried)
    fresh_wins = sum(r["won"] for r in fresh) / len(fresh)

    significant = abs(mean_delta) > 2 * stderr and stderr > 0
    marker = "*" if significant else " "

    return (
        f"{policy_id:<18} "
        f"carried={statistics.mean(carried_rewards):>6.2f}  "
        f"fresh={statistics.mean(fresh_rewards):>6.2f}  "
        f"Δ={mean_delta:>+6.2f}±{stderr:<4.2f}{marker} "
        f"slope={slope(carried_rewards):>+6.3f}/ep  "
        f"win {fresh_wins:>4.0%}→{carried_wins:<4.0%}  "
        f"| {trend}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--budget", type=float, default=1.2)
    parser.add_argument("--complexity", type=float, default=0.55)
    parser.add_argument(
        "--policies",
        type=str,
        default="contextual_bandit,mdp,markov_game,marl",
        help="Comma-separated policy ids. Baselines have nothing to carry, so they are a "
        "useful negative control if you include them.",
    )
    args = parser.parse_args()

    overrides = {
        "max_steps": args.max_steps,
        "budget_usd": args.budget,
        "task_complexity": args.complexity,
    }
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = [p for p in policies if p not in POLICY_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown policies: {', '.join(unknown)}")

    print(
        f"episodes={args.episodes} per arm, seeds {args.seed_base}..{args.seed_base + args.episodes - 1}, "
        f"max_steps={args.max_steps} budget=${args.budget} complexity={args.complexity}\n"
    )
    print(
        "Δ is the paired carried-minus-fresh reward on identical task instances. "
        "* marks |Δ| > 2 standard errors.\n"
    )

    for policy_id in policies:
        carried = run_campaign(policy_id, args.episodes, args.seed_base, True, **overrides)
        fresh = run_campaign(policy_id, args.episodes, args.seed_base, False, **overrides)
        print(summarize(policy_id, carried, fresh))

    print("\nTrend column is the mean reward of the first, middle and final third of the carried arm.")


if __name__ == "__main__":
    main()
