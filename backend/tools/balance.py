"""Offline balance harness.

Runs episodes across every policy and reports termination reasons, score distribution and
belief dynamics. This is the measurement tool for the tuning question "is a run winnable?",
and the batch half of the offline-evaluation direction in the research roadmap.

    python -m tools.balance --episodes 40
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter

from app.orchestration.engine import OrchestrationEngine, RunConfig
from app.orchestration.policies import POLICY_REGISTRY


def run_episode(policy: str, seed: int, **overrides) -> dict:
    config = RunConfig(
        task="balance probe",
        policy=policy,
        seed=seed,
        **overrides,
    )
    engine = OrchestrationEngine(config)
    steps = 0
    while not engine.done and steps < config.max_steps + 5:
        engine.step()
        steps += 1
    state = engine.state
    return {
        "reason": state.termination_reason or "none",
        "reward": engine.cumulative_reward,
        "steps": steps,
        "confidence": state.confidence,
        "entropy": state.entropy,
        "quality": state.quality,
        "verification": state.verification_score,
        "unresolved": state.unresolved_ratio,
        "cost": engine.total_cost,
    }


def summarize(policy: str, results: list[dict]) -> str:
    reasons = Counter(r["reason"] for r in results)
    wins = reasons.get("goal_reached", 0)
    rewards = [r["reward"] for r in results]
    confidences = [r["confidence"] for r in results]
    entropies = [r["entropy"] for r in results]

    top = ", ".join(f"{k}={v}" for k, v in reasons.most_common(3))
    return (
        f"{policy:<20} "
        f"win={wins / len(results):>5.0%}  "
        f"reward={statistics.mean(rewards):>7.2f}±{statistics.pstdev(rewards):<5.2f}  "
        f"conf={statistics.mean(confidences):>5.3f}  "
        f"H={statistics.mean(entropies):>5.3f}  "
        f"steps={statistics.mean(r['steps'] for r in results):>5.1f}  "
        f"| {top}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--budget", type=float, default=1.2)
    parser.add_argument("--complexity", type=float, default=0.55)
    parser.add_argument("--confidence-target", type=float, default=0.55)
    args = parser.parse_args()

    overrides = {
        "max_steps": args.max_steps,
        "budget_usd": args.budget,
        "task_complexity": args.complexity,
        "confidence_target": args.confidence_target,
    }

    print(
        f"episodes={args.episodes} max_steps={args.max_steps} budget=${args.budget} "
        f"complexity={args.complexity} confidence_target={args.confidence_target}\n"
    )

    all_results: list[dict] = []
    for policy in sorted(POLICY_REGISTRY):
        results = [run_episode(policy, seed, **overrides) for seed in range(args.episodes)]
        all_results.extend(results)
        print(summarize(policy, results))

    print("\nTermination reasons across every policy:")
    for reason, count in Counter(r["reason"] for r in all_results).most_common():
        print(f"  {reason:<20} {count / len(all_results):>5.0%}")

    finals = [r["confidence"] for r in all_results]
    print(
        f"\nFinal confidence: min={min(finals):.3f} "
        f"median={statistics.median(finals):.3f} max={max(finals):.3f}"
    )
    entropies = [r["entropy"] for r in all_results]
    print(
        f"Final entropy:    min={min(entropies):.3f} "
        f"median={statistics.median(entropies):.3f} max={max(entropies):.3f}"
    )


if __name__ == "__main__":
    main()
