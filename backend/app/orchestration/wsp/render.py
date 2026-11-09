"""Render the markov_game policy's raw weights for an LLM to read.

Every number is labelled: ``q_weights`` rows are agent ids and columns are feature names from
:data:`app.orchestration.state.FEATURE_NAMES`; ``synergy`` lists only the upper triangle
``[i, j]`` with ``i < j``, because that is the only half the policy reads (it iterates
``itertools.combinations``). Showing the lower half would invite edits that do nothing.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from ..agents import AGENT_IDS, SOLO_AGENT
from ..state import FEATURE_NAMES


def _fmt(value: float) -> str:
    return f"{value:+.3f}"


def _q_table(q: np.ndarray) -> list[str]:
    header = "| row | agent | " + " | ".join(
        f"{j}:{name}" for j, name in enumerate(FEATURE_NAMES)
    ) + " |"
    rule = "|" + "---|" * (len(FEATURE_NAMES) + 2)
    lines = [header, rule]
    for i, agent in enumerate(AGENT_IDS):
        cells = " | ".join(_fmt(float(v)) for v in q[i])
        lines.append(f"| {i} | {agent} | {cells} |")
    return lines


def _synergy_table(synergy: np.ndarray) -> list[str]:
    lines = ["| index | pair | value |", "|---|---|---|"]
    for i, j in combinations(range(len(AGENT_IDS)), 2):
        lines.append(
            f"| [{i}, {j}] | {AGENT_IDS[i]} + {AGENT_IDS[j]} | {_fmt(float(synergy[i, j]))} |"
        )
    return lines


def _terminate_table(weights: np.ndarray) -> list[str]:
    lines = ["| index | feature | value |", "|---|---|---|"]
    for j, name in enumerate(FEATURE_NAMES):
        lines.append(f"| [{j}] | {name} | {_fmt(float(weights[j]))} |")
    return lines


def _diagnostics(calibration: dict[str, Any] | None) -> list[str]:
    if not calibration:
        return ["(no calibration diagnostics)"]
    steps_to_escalate = calibration.get("mean_steps_to_escalate")
    lines = [
        (f"- calibration seeds: {calibration.get('seeds')} "
        f"({calibration.get('episodes')} sim episodes; none of these are evaluation seeds)"),
        f"- escalation rate: {calibration.get('escalation_rate', 0.0):.0%}"
        + (f", at step {steps_to_escalate:.1f} on average" if steps_to_escalate else ""),
        f"- mean steps per episode: {calibration.get('mean_steps', 0.0):.1f}",
        f"- termination reasons: {calibration.get('termination_reasons')}",
        (f"- coalition sizes (steps): {calibration.get('coalition_sizes')}; "
        f"{calibration.get('coalition_steps_per_episode', 0.0):.1f} coalition steps per episode"),
        "- top coalitions: "
        + (", ".join(f"{c['agents']}×{c['count']}" for c in calibration.get("top_coalitions", []))
           or "none"),
        (f"- mean cost: ${calibration.get('mean_cost_usd', 0.0):.4f}; "
        f"mean tokens: {calibration.get('mean_tokens', 0.0):.0f}"),
    ]
    return lines


def render_policy(
    policy: Any,
    calibration: dict[str, Any] | None = None,
    limits: Any | None = None,
) -> str:
    """Markdown description of ``policy``'s weights, how they are used, and how it behaves."""
    q = np.asarray(policy.q_weights, dtype=float)
    synergy = np.asarray(policy.synergy, dtype=float)
    terminate = np.asarray(policy.terminate_weights, dtype=float)
    total = q.size + synergy.size + terminate.size
    solo = AGENT_IDS.index(SOLO_AGENT)

    sections = [
        "# Cooperative Markov game policy — raw weights",
        "",
        "## How the weights are used",
        "",
        "phi(s) is the feature vector below (feature 0 is a constant bias of 1; the rest are",
        "in [0, 1]; `has_escalated` is a 0/1 flag and `stall` saturates at 1 after three",
        "steps without progress). Actions are sampled from a",
        f"softmax over scores at temperature {getattr(policy, 'temperature', 0.26):.2f}.",
        "",
        "- Q_i(s) = q_weights[i] · phi(s), one row per agent.",
        "- Coalition C (2-3 specialists; the generalist is never in a coalition):",
        ("  V(s, C) = Σ_{i∈C} Q_i(s) + Σ_{i<j∈C} synergy[i, j]"
        f" − {getattr(policy, 'coalition_cost', 0.35):.2f}·(|C|−1)·cost_pressure(s),"),
        "  cost_pressure = 1 − 0.5·budget_remaining − 0.5·latency_remaining.",
        "  RUN_PARALLEL scores the best such coalition.",
        "- TERMINATE = terminate_weights · phi(s) + readiness prior, where the prior is",
        ("  −1.6 + 3.4·confidence·verification_score·(1 − unresolved_ratio)"
        " + 1.8·(1 − budget_remaining)."),
        f"- Before escalation only INVOKE_GENERALIST (row {solo}), ESCALATE and TERMINATE are",
        "  legal. ESCALATE always scores 0, so the generalist row alone decides when to",
        "  escalate: a high Q_generalist keeps the run solo; a negative one escalates.",
        "- The policy keeps learning online during an episode; your edit sets its starting point.",
        "",
        f"## q_weights — shape {list(q.shape)}; edit with index [row, column]",
        "",
        *_q_table(q),
        "",
        f"## synergy — shape {list(synergy.shape)}; only [i, j] with i < j is read",
        "",
        *_synergy_table(synergy),
        "",
        f"## terminate_weights — shape {list(terminate.shape)}; edit with index [column]",
        "",
        *_terminate_table(terminate),
        "",
        "## Calibration diagnostics (behaviour of these exact weights)",
        "",
        *_diagnostics(calibration),
    ]
    if limits is not None:
        max_changed = int(getattr(limits, "max_changed_fraction", 0.0) * total)
        sections += [
            "",
            "## Edit limits (a guard rejects the whole patch if any is violated)",
            "",
            f"- |new value| ≤ {getattr(limits, 'max_abs_value', '?')}",
            f"- |new value − current value| ≤ {getattr(limits, 'max_abs_delta', '?')}",
            (f"- at most {getattr(limits, 'max_changed_fraction', 0.0):.0%} of the {total}"
            f" parameters may change (≤ {max_changed} entries)"),
        ]
    return "\n".join(sections) + "\n"
