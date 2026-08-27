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


def say(line: str = "") -> None:
    # Redirected output is block-buffered, and an experiment runs for many minutes; without
    # flushing, the log stays empty until the very end.
    print(line, flush=True)


def log(kind: str, payload: dict) -> None:
    head = f"[{payload.get('arm', '-'):>10} s{payload.get('seed', '-')}]"
    if kind == "run_created":
        say(f"{head} run {payload['run_id']}  watch {payload['watch']}")
    elif kind == "step_open":
        say(f"{head} step {payload['step']} {payload['action']} -> {', '.join(payload['agents'])}")
    elif kind == "agent_done":
        flag = "" if payload["parsed"] else "  [no verdict block]"
        say(
            f"{head}   {payload['agent']:<12} {payload['model']:<18} "
            f"{payload['outcome']:<8} conf {payload['confidence']:.2f}  "
            f"new {payload['new_tokens']:>6} / cached {payload['cached_tokens']:>6}"
            f"{flag}"
        )
    elif kind == "step_done":
        say(
            f"{head} reward {payload['reward']:+.3f}  entropy {payload['entropy']:.3f}"
            f"{'  DONE' if payload['done'] else ''}"
        )
    elif kind == "open_refused":
        say(f"{head} {payload['detail']}")
    elif kind == "step_lost":
        say(
            f"{head} step {payload['step']} lost: {payload['detail']}"
            f"  ({payload['retries_left']} retries left)"
        )
    elif kind == "agent_stalled":
        say(f"{head}   {payload['agent']} repeated itself at step {payload['step']}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--hypothesis", action="append", dest="hypotheses", required=True)
    parser.add_argument("--rubric-file", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105])
    parser.add_argument(
        "--arm",
        action="append",
        dest="arms",
        default=None,
        help=(
            "Repeatable. 'control' is mandatory. The arena's own arms are cascade and "
            "always_orchestrate; maf_sequential, maf_concurrent and maf_handoff are driven by "
            "Microsoft Agent Framework patterns instead. Defaults to control and cascade."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--budget", type=float, default=250_000.0, help="in tokens")
    parser.add_argument(
        "--latency-budget-ms",
        type=float,
        default=1_800_000.0,
        help=(
            "Matched across arms. The API default of 90s was sized for sampled timings and a "
            "real call takes 10-25s, so leaving it would end every run on latency_exhausted "
            "after four steps rather than on its own terms."
        ),
    )
    parser.add_argument("--orchestrator-model", default="claude-opus-5")
    parser.add_argument("--worker-model", default="claude-haiku-4.5")
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument(
        "--agent-timeout-s",
        type=float,
        default=300.0,
        help="A haiku call takes 10-25s, so a wait this long means the session has hung.",
    )
    args = parser.parse_args()

    arms = args.arms or [CONTROL, "cascade"]
    if CONTROL not in arms:
        say(f"error: every experiment needs the '{CONTROL}' arm, or it is not a comparison.")
        return 2
    if len(args.hypotheses) < 3:
        say(
            "error: at least three competing hypotheses. Four is better — four rephrasings of "
            "one answer collapse the belief trivially and produce a result that is not real."
        )
        return 2
    if len(args.seeds) < 5:
        print(
            f"warning: {len(args.seeds)} seed(s) is a smoke test, not a result. "
            "A win rate needs |p-0.5| > 1/sqrt(n), which 5 seeds can only just reach.",
            file=sys.stderr,
            flush=True,
        )

    rubric = args.rubric_file.read_text(encoding="utf-8").strip()
    if not rubric:
        say("error: the rubric is empty. Write it before running, not after seeing results.")
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
                say(f"\n=== {arm}  seed {seed} ===")
                try:
                    result = await drive_arm(
                        arena,
                        arm=arm,
                        strategy=arm,
                        seed=seed,
                        task=args.task,
                        experiment=args.experiment,
                        hypotheses=args.hypotheses,
                        max_steps=args.max_steps,
                        budget=args.budget,
                        latency_budget_ms=args.latency_budget_ms,
                        default_model=args.orchestrator_model,
                        agent_models=agent_models,
                        agent_timeout_s=args.agent_timeout_s,
                        on_event=log,
                    )
                except Exception as exc:  # noqa: BLE001 - one arm must not cost the rest
                    say(f"    ABANDONED: {type(exc).__name__}: {exc}")
                    continue
                results[seed][arm] = result
                say(
                    f"    {result.steps} steps, driven by {result.driver}, "
                    f"{result.new_tokens:,} fresh tokens, ended on {result.terminated_reason}"
                    f"{f',  {result.stalled_reports} stalled' if result.stalled_reports else ''}"
                    f"{'' if result.complete else '  [INCOMPLETE]'}"
                )

        say("\n=== blind pairwise judging ===")
        judge = Judge(args.judge_model)
        challengers = [a for a in arms if a != CONTROL]
        for seed, by_arm in results.items():
            # A seed is only a paired comparison if every arm actually finished it. Judging a
            # truncated answer against a complete one measures the outage, not the pattern.
            missing = [a for a in arms if a not in by_arm]
            truncated = [a for a, r in by_arm.items() if not r.complete]
            if missing or truncated:
                say(
                    f"  seed {seed}: not judged — "
                    f"{', '.join(missing + truncated)} did not finish"
                )
                continue

            control = by_arm[CONTROL]
            for arm in challengers:
                other = by_arm[arm]
                if not control.final_answer and not other.final_answer:
                    say(f"  seed {seed} {arm}: both arms produced nothing, skipping")
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
                say(f"  seed {seed}  {CONTROL} vs {arm} -> {named}  (shown first: {shown})")

        say("\n=== comparison ===")
        try:
            comparison = arena.comparison(args.experiment)
        except ArenaError as exc:
            say(f"  {exc}")
            return 1

        say(f"  {comparison['verdict']}")
        unit = comparison["cost_unit"]
        for arm in comparison["arms"]:
            quality = arm["mean_quality"]
            say(
                f"  {arm['arm']:<20} runs {arm['runs']}  "
                f"esc {arm['escalated']}/{arm['runs']}  "
                f"quality {'unjudged' if quality is None else f'{quality:.2f}'}  "
                f"cost {arm['mean_cost_usd']:,.0f} {unit}  "
                f"tokens {arm['mean_tokens']:,.0f}"
            )
        for caveat in comparison["caveats"]:
            say(f"  ! {caveat}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
