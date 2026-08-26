"""Blind pairwise judging in a session that never saw either answer produced.

The existing CLI has Copilot judge answers it wrote itself minutes earlier, which is a conflict
of interest no rubric survives. A fresh agent has no such memory: it receives two anonymous
answers, in an order shuffled per seed, and cannot tell which arm wrote which.

Forced choice rather than absolute scoring, for the reason MT-Bench and Chatbot Arena settled
on it — a judge scoring one answer at a time drifts to the top of the range, and two arms come
back 0.96 against 0.98, which is noise between two scorings rather than a difference.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from agent_framework.github import GitHubCopilotAgent, GitHubCopilotOptions

VERDICT = re.compile(r"\b(?:winner|verdict)\s*[:=]\s*\**\s*(a|b|tie)\b", re.IGNORECASE)

JUDGE_INSTRUCTIONS = """You judge two anonymous answers to the same task against a rubric.

You do not know who wrote either one, and you must not speculate. Length, confidence and
formatting are not quality. Judge only what the rubric asks for.

`tie` is a real verdict, not a failure to decide. If neither answer is better against the
rubric, say tie — that is evidence the two approaches are indistinguishable, which is a finding.
Manufacturing a preference to avoid a tie invents a result.

End your reply with exactly one line:

VERDICT: A
VERDICT: B
VERDICT: TIE
"""


@dataclass
class Judgement:
    winner: str
    """``a`` or ``b`` in the *caller's* labelling, or ``tie``."""
    notes: str
    presented_first: str


class Judge:
    def __init__(self, model: str, timeout: float = 600.0) -> None:
        self._agent = GitHubCopilotAgent(
            name="judge",
            instructions=JUDGE_INSTRUCTIONS,
            default_options=GitHubCopilotOptions(model=model, timeout=timeout),
        )
        self.model = model

    async def compare(
        self, *, task: str, rubric: str, answer_a: str, answer_b: str, seed: int
    ) -> Judgement:
        # Shuffle per seed so position bias cannot line up with an arm across the experiment.
        flip = random.Random(seed).random() < 0.5
        first, second = (answer_b, answer_a) if flip else (answer_a, answer_b)

        prompt = (
            f"TASK\n{task}\n\n"
            f"RUBRIC\n{rubric}\n\n"
            f"--- ANSWER A ---\n{first or '(this arm produced nothing)'}\n\n"
            f"--- ANSWER B ---\n{second or '(this arm produced nothing)'}\n\n"
            "Which better satisfies the rubric? Give your reasoning, then the verdict line."
        )
        # No session: each comparison is independent, and carrying history between seeds would
        # let an earlier verdict anchor a later one.
        response = await self._agent.run(prompt)
        text = (response.text or "").strip()

        match = VERDICT.search(text)
        shown = (match.group(1).lower() if match else "tie")
        if shown == "tie":
            winner = "tie"
        elif flip:
            winner = "b" if shown == "a" else "a"
        else:
            winner = shown

        return Judgement(
            winner=winner,
            notes=text[-1200:],
            presented_first="b" if flip else "a",
        )
