"""Experiment comparison: does orchestration beat a single agent on your task?

Two rules drive the design.

**Pair on seeds.** Arms that ran the same seed saw the same task instance, so the difference
between them is exact rather than confounded by which tasks each happened to draw. Unpaired
means across differently-seeded arms would mostly measure luck. This mirrors what
``tools/campaign.py`` already does for carried-versus-fresh.

**Never rank on internal reward.** The reward function pays for belief collapse and subtask
resolution, both of which presume decomposition, so a single-agent control scores badly on it
regardless of answer quality. Ranking arms on it would let orchestration win by construction.
Reward is reported per arm and excluded from every comparison; quality comes from
``RunVerdict``, which is judged outside this module.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import PairwiseVerdict, Run, RunVerdict

CONTROL_ARM = "control"

# Lower is better for each of these; quality is handled separately.
COST_METRICS: tuple[str, ...] = ("cost_usd", "latency_ms", "tokens", "steps")


class ExperimentService:
    def __init__(self, session: Session) -> None:
        self.session = session

    # -------------------------------------------------------------- listing
    def list_experiments(self) -> list[dict]:
        grouped: dict[str, list[Run]] = defaultdict(list)
        for run in self._tagged_runs():
            grouped[str(run.config.get("experiment"))].append(run)

        return [
            {
                "experiment": name,
                "arms": sorted({self._arm_of(r) for r in members}),
                "runs": len(members),
                "seeds": sorted({r.seed for r in members}),
                "tasks": sorted({r.task for r in members}),
                "has_control": any(self._arm_of(r) == CONTROL_ARM for r in members),
            }
            for name, members in sorted(grouped.items())
        ]

    # ----------------------------------------------------------- comparison
    def compare(self, experiment: str) -> dict:
        runs = [
            run for run in self._tagged_runs() if run.config.get("experiment") == experiment
        ]
        if not runs:
            raise LookupError(f"No runs tagged with experiment '{experiment}'")

        verdicts = self._verdicts_for([r.id for r in runs])
        by_arm: dict[str, list[Run]] = defaultdict(list)
        for run in runs:
            by_arm[self._arm_of(run)].append(run)

        arms = [
            self._summarize(arm, members, verdicts) for arm, members in sorted(by_arm.items())
        ]
        control = by_arm.get(CONTROL_ARM)

        arm_of_run = {r.id: self._arm_of(r) for r in runs}
        pairwise = self._pairwise(experiment, arm_of_run)
        for arm in arms:
            arm["pairwise"] = pairwise.get(arm["arm"])

        for arm in arms:
            if control is None or arm["arm"] == CONTROL_ARM:
                arm["vs_control"] = None
            else:
                arm["vs_control"] = self._paired_delta(by_arm[arm["arm"]], control, verdicts)

        return {
            "experiment": experiment,
            "tasks": sorted({run.task for run in runs}),
            "control_arm": CONTROL_ARM if control else None,
            "arms": arms,
            "verdict": self._headline(arms),
            "caveats": self._caveats(arms, control),
        }

    def _pairwise(self, experiment: str, arm_of_run: dict[str, str]) -> dict[str, dict]:
        """Win/loss/tie per arm from blind head-to-head comparisons."""
        rows = self.session.scalars(
            select(PairwiseVerdict).where(PairwiseVerdict.experiment == experiment)
        ).all()
        if not rows:
            return {}

        tally: dict[str, dict[str, int]] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "ties": 0}
        )
        for row in rows:
            arm_a = arm_of_run.get(row.run_a_id)
            arm_b = arm_of_run.get(row.run_b_id)
            if not arm_a or not arm_b or arm_a == arm_b:
                continue
            if row.winner == "tie":
                tally[arm_a]["ties"] += 1
                tally[arm_b]["ties"] += 1
            elif row.winner == "a":
                tally[arm_a]["wins"] += 1
                tally[arm_b]["losses"] += 1
            else:
                tally[arm_b]["wins"] += 1
                tally[arm_a]["losses"] += 1

        out: dict[str, dict] = {}
        for arm, counts in tally.items():
            decisive = counts["wins"] + counts["losses"]
            total = decisive + counts["ties"]
            rate = counts["wins"] / decisive if decisive else None
            # Against a fair coin the standard error of the win rate is 0.5/sqrt(n), so two of
            # those is 1/sqrt(n) — the same two-standard-error bar used elsewhere here.
            significant = bool(
                decisive >= 2 and rate is not None and abs(rate - 0.5) > 1.0 / (decisive**0.5)
            )
            out[arm] = {
                **counts,
                "comparisons": total,
                "decisive": decisive,
                "win_rate": round(rate, 4) if rate is not None else None,
                "significant": significant,
            }
        return out

    # ------------------------------------------------------------ internals
    def _tagged_runs(self) -> list[Run]:
        runs = self.session.scalars(select(Run).order_by(Run.created_at)).unique().all()
        return [run for run in runs if (run.config or {}).get("experiment")]

    @staticmethod
    def _arm_of(run: Run) -> str:
        return str((run.config or {}).get("arm") or run.policy)

    def _verdicts_for(self, run_ids: list[str]) -> dict[str, float]:
        if not run_ids:
            return {}
        rows = self.session.scalars(
            select(RunVerdict).where(RunVerdict.run_id.in_(run_ids))
        ).all()
        scores: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            scores[row.run_id].append(row.score)
        return {run_id: statistics.mean(vals) for run_id, vals in scores.items()}

    @staticmethod
    def _metrics(run: Run) -> dict[str, float]:
        return {
            "cost_usd": run.total_cost,
            "latency_ms": run.total_latency_ms,
            "tokens": float(run.total_tokens),
            "steps": float(run.step_count),
        }

    def _summarize(self, arm: str, members: list[Run], verdicts: dict[str, float]) -> dict:
        n = max(len(members), 1)
        scored = [verdicts[r.id] for r in members if r.id in verdicts]
        summary = {
            "arm": arm,
            "policy": members[0].policy,
            "runs": len(members),
            "seeds": sorted({r.seed for r in members}),
            "goal_reached": sum(1 for r in members if r.termination_reason == "goal_reached"),
            "escalated": sum(1 for r in members if (r.current_state or {}).get("has_escalated")),
            "mean_quality": statistics.mean(scored) if scored else None,
            "judged_runs": len(scored),
            "mean_internal_reward": sum(r.cumulative_reward for r in members) / n,
            "run_ids": [r.id for r in members],
        }
        for key in COST_METRICS:
            summary[f"mean_{key}"] = sum(self._metrics(r)[key] for r in members) / n
        return summary

    def _paired_delta(
        self, arm_runs: list[Run], control_runs: list[Run], verdicts: dict[str, float]
    ) -> dict:
        """Difference against the control on shared seeds only."""
        control_by_seed = {r.seed: r for r in control_runs}
        pairs = [(r, control_by_seed[r.seed]) for r in arm_runs if r.seed in control_by_seed]

        if not pairs:
            return {
                "paired_seeds": 0,
                "note": "No seeds shared with the control arm; run both arms on the same seeds.",
            }

        out: dict = {"paired_seeds": len(pairs)}
        for key in COST_METRICS:
            deltas = [self._metrics(a)[key] - self._metrics(c)[key] for a, c in pairs]
            stat = self._stat(deltas)
            base = statistics.mean(self._metrics(c)[key] for _, c in pairs)
            treatment = statistics.mean(self._metrics(a)[key] for a, _ in pairs)
            stat["multiple"] = round(treatment / base, 3) if base else None
            out[key] = stat

        quality = [
            verdicts[a.id] - verdicts[c.id]
            for a, c in pairs
            if a.id in verdicts and c.id in verdicts
        ]
        out["quality"] = self._stat(quality) if quality else None
        return out

    @staticmethod
    def _stat(deltas: list[float]) -> dict:
        mean = statistics.mean(deltas)
        stderr = statistics.pstdev(deltas) / max(len(deltas) ** 0.5, 1e-9)
        return {
            "mean_delta": round(mean, 6),
            "stderr": round(stderr, 6),
            "significant": bool(abs(mean) > 2 * stderr and stderr > 0),
            "n": len(deltas),
        }

    @staticmethod
    def _headline(arms: list[dict]) -> str:
        """One sentence a human can act on, or an honest refusal to give one."""
        judged = [a for a in arms if a["mean_quality"] is not None]
        control = next((a for a in arms if a["arm"] == CONTROL_ARM), None)

        if control is None:
            return "No verdict: this experiment has no control arm to compare against."

        # An arm that never escalated is the control wearing a different label.
        inert = [
            a["arm"]
            for a in arms
            if a["arm"] != CONTROL_ARM and a["escalated"] == 0 and a["runs"] > 0
        ]
        if inert and len(inert) == len(arms) - 1:
            return (
                f"No verdict: {', '.join(inert)} never escalated, so every arm ran the same solo "
                "agent as the control. This is not a comparison — give the arms more steps, or "
                "pick a strategy that escalates unconditionally."
            )

        # Pairwise preference discriminates where absolute scores compress, so it wins when present.
        contested = [
            a for a in arms if a["arm"] != CONTROL_ARM and (a.get("pairwise") or {}).get("decisive")
        ]
        if contested:
            best = max(contested, key=lambda a: a["pairwise"]["win_rate"] or 0.0)
            pw = best["pairwise"]
            multiple = ((best["vs_control"] or {}).get("cost_usd") or {}).get("multiple")
            cost_note = f" at {multiple}x the cost" if multiple else ""
            if not pw["significant"]:
                return (
                    f"No verdict: '{best['arm']}' won {pw['wins']} of {pw['decisive']} blind "
                    "comparisons, which a fair coin would produce often enough. Judge more pairs."
                )
            if pw["win_rate"] > 0.5:
                return (
                    f"'{best['arm']}' beat the control in {pw['wins']} of {pw['decisive']} blind "
                    f"comparisons ({pw['win_rate']:.0%}){cost_note}."
                )
            return (
                f"The control won: '{best['arm']}' took only {pw['wins']} of {pw['decisive']} "
                "blind comparisons. Orchestration is not paying for itself here."
            )

        if len(judged) < 2:
            return (
                "No verdict: quality has not been judged on enough arms. Cost is measured, but "
                "cheaper only counts as better if the answer held up. Score the runs first."
            )

        best = max(judged, key=lambda a: a["mean_quality"])
        if best["arm"] == CONTROL_ARM:
            return (
                f"The single-agent control scored highest on quality "
                f"({control['mean_quality']:.2f}). Orchestration is not paying for itself here."
            )

        delta = (best["vs_control"] or {}).get("quality")
        if not delta or not delta["significant"]:
            return (
                f"'{best['arm']}' leads on quality, but the paired difference is within noise. "
                "Add seeds, or accept that the arms are indistinguishable."
            )
        multiple = ((best["vs_control"] or {}).get("cost_usd") or {}).get("multiple")
        return (
            f"'{best['arm']}' beats the control on quality by {delta['mean_delta']:+.2f} "
            f"(more than 2 standard errors) at {multiple}x the cost."
        )

    @staticmethod
    def _caveats(arms: list[dict], control: list[Run] | None) -> list[str]:
        notes: list[str] = []
        if control is None:
            notes.append(
                "No arm is named 'control'. Without a single-agent baseline there is nothing to "
                "attribute a difference to."
            )

        # An arm that never escalated did the same thing the control did, so any difference
        # between them is noise between two scorings of identical work.
        inert = [
            a["arm"]
            for a in arms
            if a["arm"] != CONTROL_ARM and a["escalated"] == 0 and a["runs"] > 0
        ]
        if inert:
            notes.append(
                f"NOT A COMPARISON: {', '.join(inert)} never escalated, so those arms ran the "
                "same solo agent as the control. Give them more steps, or use a strategy that "
                "escalates unconditionally, before reading anything into the difference."
            )

        thin = [a["arm"] for a in arms if len(a["seeds"]) < 5]
        if thin:
            notes.append(
                f"Thin evidence on {', '.join(thin)}: fewer than 5 seeds, so paired differences "
                "are unlikely to clear two standard errors."
            )
        unjudged = [a["arm"] for a in arms if a["mean_quality"] is None]
        if unjudged:
            notes.append(
                f"Unjudged arms: {', '.join(unjudged)}. Cost without quality cannot tell you "
                "whether orchestration was worth it."
            )

        if not any((a.get("pairwise") or {}).get("decisive") for a in arms):
            notes.append(
                "No blind pairwise comparisons recorded. Absolute scores compress — judges "
                "rating one answer at a time drift to the top of the range — so prefer "
                "POST /api/experiments/{name}/pairwise for the quality question."
            )

        notes.append(
            "Internal reward is reported per arm but never compared: it pays for belief "
            "collapse, which a single-agent control does not attempt."
        )
        return notes
