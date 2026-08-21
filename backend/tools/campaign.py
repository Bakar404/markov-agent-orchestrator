"""Cross-episode learning harness (CLI).

Thin wrapper over ``app.services.campaign_service`` so the terminal and the UI report the same
numbers from the same implementation.

    python -m tools.campaign --episodes 40 --policies mdp,markov_game,marl
"""

from __future__ import annotations

import argparse

from app.orchestration.policies import POLICY_REGISTRY
from app.services.campaign_service import run_campaign


def format_row(entry: dict) -> str:
    carried = entry["carried"]
    fresh = entry["fresh"]
    marker = "*" if entry["significant"] else " "
    trend = " -> ".join(f"{b:+.2f}" for b in entry["blocks"])

    return (
        f"{entry['policy']:<18} "
        f"carried={carried['mean_reward']:>6.2f}  "
        f"fresh={fresh['mean_reward']:>6.2f}  "
        f"D={entry['delta']:>+6.2f}+-{entry['stderr']:<4.2f}{marker} "
        f"slope={entry['slope']:>+6.3f}/ep  "
        f"win {fresh['win_rate']:>4.0%}->{carried['win_rate']:<4.0%}  "
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
        help="Comma-separated policy ids. Baselines carry no state, so including one gives a "
        "negative control that must report exactly zero.",
    )
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = [p for p in policies if p not in POLICY_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown policies: {', '.join(unknown)}")

    outcome = run_campaign(
        policies=policies,
        episodes=args.episodes,
        seed_base=args.seed_base,
        max_steps=args.max_steps,
        budget_usd=args.budget,
        task_complexity=args.complexity,
    )
    config = outcome["config"]

    print(
        f"episodes={config['episodes']} per arm, "
        f"seeds {config['seed_base']}..{config['seed_base'] + config['episodes'] - 1}, "
        f"max_steps={config['max_steps']} budget=${config['budget_usd']} "
        f"complexity={config['task_complexity']}\n"
    )
    print(
        "D is the paired carried-minus-fresh reward on identical task instances. "
        "* marks |D| > 2 standard errors.\n"
    )

    for entry in outcome["results"]:
        print(format_row(entry))

    print("\nTrend column is the mean reward of the first, middle and final third of the carried arm.")


if __name__ == "__main__":
    main()
