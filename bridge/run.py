"""Run a full experiment: every arm on every seed, then blind pairwise judging.

    C:\\venvs\\arena-maf\\Scripts\\python bridge/run.py \\
        --experiment coord-cost \\
        --task "..." \\
        --hypothesis "..." --hypothesis "..." --hypothesis "..." --hypothesis "..." \\
        --rubric-file rubric.md \\
        --seeds 101 102 103 104 105

Five seeds is the floor, not a suggestion. A win rate needs |p - 0.5| > 1/sqrt(n) to clear two
standard errors, so four seeds cannot produce a significant result even if one arm sweeps.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from arena_bridge.arena import Arena, ArenaError  # noqa: E402
from arena_bridge.driver import ArmResult, drive_arm  # noqa: E402
from arena_bridge.judge import Judge  # noqa: E402

CONTROL = "control"

# The arena's caveats are written with em-dashes, which land as mojibake on a cp437 console.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def log(kind: str, payload: dict) -> None:
    head = f"[{payload.get('arm', '-'):>10} s{payload.get('seed', '-')}]"
    if kind == "run_created":
        print(f"{head} run {payload['run_id']}  watch {payload['watch']}")
    elif kind == "step_open":
        print(f"{head} step {payload['step']} {payload['action']} -> {', '.join(payload['agents'])}")
    elif kind == "agent_done":
        flag = "" if payload["parsed"] else "  [no verdict block]"
        print(
            f"{head}   {payload['agent']:<12} {payload['model']:<18} "
            f"{payload['outcome']:<8} conf {payload['confidence']:.2f}  "
            f"new {payload['new_tokens']:>6} / cached {payload['cached_tokens']:>6}"
            f"{flag}"
        )
    elif kind == "step_done":
        print(
            f"{head} reward {payload['reward']:+.3f}  entropy {payload['entropy']:.3f}"
            f"{'  DONE' if payload['done'] else ''}"
        )
    elif kind == "open_refused":
        print(f"{head} {payload['detail']}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--hypothesis", action="append", dest="hypotheses", required=True)
    parser.add_argument("--rubric-file", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105])
    parser.add_argument("--arm", action="append", dest="arms", default=None)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--budget", type=float, default=250_000.0, help="in tokens")
    parser.add_argument("--orchestrator-model", default="claude-opus-5")
    parser.add_argument("--worker-model", default="claude-haiku-4.5")
    parser.add_argument("--judge-model", default="gpt-5.4")
    args = parser.parse_args()

    arms = args.arms or [CONTROL, "cascade"]
    if CONTROL not in arms:
        print(f"error: every experiment needs the '{CONTROL}' arm, or it is not a comparison.")
        return 2
    if len(args.hypotheses) < 3:
        print(
            "error: at least three competing hypotheses. Four is better — four rephrasings of "
            "one answer collapse the belief trivially and produce a result that is not real."
        )
        return 2
    if len(args.seeds) < 5:
        print(
            f"warning: {len(args.seeds)} seed(s) is a smoke test, not a result. "
            "A win rate needs |p-0.5| > 1/sqrt(n), which 5 seeds can only just reach.",
            file=sys.stderr,
        )

    rubric = args.rubric_file.read_text(encoding="utf-8").strip()
    if not rubric:
        print("error: the rubric is empty. Write it before running, not after seeing results.")
        return 2

    # Workers stay cheap so the arms differ in how they spend rather than in how much.
    agent_models = {
        agent: args.worker_model
        for agent in ("researcher", "critic", "verifier", "memory", "executor")
    }

    with Arena(args.api) as arena:
        results: dict[int, dict[str, ArmResult]] = {}
        for seed in args.seeds:
            results[seed] = {}
            # Control first, so a solo answer is never contaminated by having already seen
            # the specialists do the work.
            for arm in sorted(arms, key=lambda a: a != CONTROL):
                print(f"\n=== {arm}  seed {seed} ===")
                results[seed][arm] = await drive_arm(
                    arena,
                    arm=arm,
                    strategy=arm,
                    seed=seed,
                    task=args.task,
                    experiment=args.experiment,
                    hypotheses=args.hypotheses,
                    max_steps=args.max_steps,
                    budget=args.budget,
                    default_model=args.orchestrator_model,
                    agent_models=agent_models,
                    on_event=log,
                )

        print("\n=== blind pairwise judging ===")
        judge = Judge(args.judge_model)
        challengers = [a for a in arms if a != CONTROL]
        for seed, by_arm in results.items():
            control = by_arm[CONTROL]
            for arm in challengers:
                other = by_arm[arm]
                if not control.final_answer and not other.final_answer:
                    print(f"  seed {seed} {arm}: both arms produced nothing, skipping")
                    continue
                judgement = await judge.compare(
                    task=args.task,
                    rubric=rubric,
                    answer_a=control.final_answer,
                    answer_b=other.final_answer,
                    seed=seed,
                )
                arena.record_pairwise(
                    args.experiment,
                    run_a=control.run_id,
                    run_b=other.run_id,
                    winner=judgement.winner,
                    judge=f"maf:{args.judge_model}",
                    rubric=rubric,
                    notes=judgement.notes,
                )
                shown = {"a": CONTROL, "b": arm}[judgement.presented_first]
                named = {"a": CONTROL, "b": arm, "tie": "tie"}[judgement.winner]
                print(f"  seed {seed}  {CONTROL} vs {arm} -> {named}  (shown first: {shown})")

        print("\n=== comparison ===")
        try:
            comparison = arena.comparison(args.experiment)
        except ArenaError as exc:
            print(f"  {exc}")
            return 1

        print(f"  {comparison['verdict']}")
        unit = comparison["cost_unit"]
        for arm in comparison["arms"]:
            quality = arm["mean_quality"]
            print(
                f"  {arm['arm']:<20} runs {arm['runs']}  "
                f"esc {arm['escalated']}/{arm['runs']}  "
                f"quality {'unjudged' if quality is None else f'{quality:.2f}'}  "
                f"cost {arm['mean_cost_usd']:,.0f} {unit}  "
                f"tokens {arm['mean_tokens']:,.0f}"
            )
        for caveat in comparison["caveats"]:
            print(f"  ! {caveat}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
