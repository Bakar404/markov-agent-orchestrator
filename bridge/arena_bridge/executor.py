"""Run one agent brief through Microsoft Agent Framework and measure what it actually cost.

Every Copilot CLI invocation carries roughly 15,800 tokens of fixed session context — system
prompt, tool definitions, skills — before the agent does any work. Measured on this machine, a
turn that opens a fresh session processes about 9,300 new tokens; a turn continuing an existing
one processes about 280. Same model, same question, 34x apart.

That gap is the whole point. ``total_token_count`` sits near 16,000 either way, because it
counts cache reads at face value, so reporting it would make every arm look identical. What
separates them is how many tokens the provider had to process *fresh*, which is what a
delegation to a new specialist costs and a continued conversation does not.

So the pool keeps one session per agent id for the life of a run. A solo arm returning to the
same agent amortises its context; an arm that fans out to specialists pays to establish each
one. Neither is penalised by construction — the difference falls out of what was measured.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from agent_framework.github import GitHubCopilotAgent, GitHubCopilotOptions

JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
BARE_OBJECT = re.compile(r"(\{[^{}]*\"outcome\"[^{}]*\})", re.DOTALL)

REPORT_CONTRACT = """
When you have finished, end your reply with a fenced json block in exactly this shape:

```json
{
  "outcome": "success | partial | failure",
  "confidence": 0.0,
  "claimed_hypothesis": 0,
  "summary": "one line, under 160 characters"
}
```

Report it honestly, because these numbers drive the belief update rather than decorate it:

- `outcome` is `success` only if you fulfilled the brief. Use `partial` for genuine progress and
  `failure` when you could not do it or got it wrong. A failure is useful data.
- `confidence` is calibrated, not aspirational.
- `claimed_hypothesis` is the index your work actually supports. **Omit the field entirely if
  your work supported none of them.** There is no hidden answer key here; belief mass follows
  what agents argue for, and confidence only rises when independent agents agree. Guessing an
  index defeats that.
"""


@dataclass
class Invocation:
    """One agent response plus the numbers read off it. Nothing here is estimated."""

    agent_id: str
    text: str
    outcome: str
    confidence: float
    summary: str
    claimed_hypothesis: int | None
    total_tokens: int
    new_tokens: int
    """Tokens the provider processed fresh: input, less cache reads, plus output.

    This is the cost figure. It separates establishing a context from continuing one."""
    cached_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    parsed: bool
    """False when the agent did not return the json block, which is reported rather than hidden."""


@dataclass
class AgentPool:
    """One agent and one session per agent id, held for the life of a run."""

    default_model: str
    timeout: float = 600.0
    _agents: dict[str, tuple[GitHubCopilotAgent, Any, str]] = field(default_factory=dict)

    def _agent_for(self, brief: dict) -> tuple[GitHubCopilotAgent, Any, str]:
        agent_id = brief["agent_id"]
        existing = self._agents.get(agent_id)
        if existing is not None:
            return existing

        model = brief.get("model") or self.default_model
        agent = GitHubCopilotAgent(
            name=agent_id,
            instructions=(
                f"You are the {brief['label']} ({brief['role']}) on a multi-agent team.\n"
                f"{brief['instruction']}\n\n"
                "Do the work. Do not describe how you would do it, and do not restate the brief."
            ),
            default_options=GitHubCopilotOptions(model=model, timeout=self.timeout),
        )
        entry = (agent, agent.create_session(), model)
        self._agents[agent_id] = entry
        return entry

    async def invoke(self, brief: dict) -> Invocation:
        agent, session, model = self._agent_for(brief)
        started = time.perf_counter()
        response = await agent.run(_prompt_for(brief), session=session)
        latency_ms = (time.perf_counter() - started) * 1000.0

        usage = dict(response.usage_details or {})
        total = int(usage.get("total_token_count") or 0)
        cached = int(usage.get("cache_read_input_token_count") or 0)
        output = int(usage.get("output_token_count") or 0)
        new_tokens = max(int(usage.get("input_token_count") or 0) - cached, 0) + output

        text = (response.text or "").strip()
        verdict, parsed = _parse_verdict(text)

        return Invocation(
            agent_id=brief["agent_id"],
            text=_without_json_block(text),
            outcome=verdict["outcome"],
            confidence=verdict["confidence"],
            summary=verdict["summary"] or f"{brief['label']}: {verdict['outcome']}",
            claimed_hypothesis=verdict["claimed_hypothesis"],
            total_tokens=max(total, 1),
            new_tokens=max(new_tokens, 1),
            cached_tokens=cached,
            output_tokens=output,
            latency_ms=latency_ms,
            model=model,
            parsed=parsed,
        )

    def to_report(self, invocation: Invocation) -> dict:
        """Shape one invocation the way ``live/report`` wants it.

        ``cost_usd`` carries new tokens because the run declares ``cost_unit="tokens"``. The
        field name predates the unit being recorded and is left alone rather than migrated
        across the database, the API and the frontend at once.
        """
        report = {
            "agent_id": invocation.agent_id,
            "outcome": invocation.outcome,
            "confidence": invocation.confidence,
            "response": invocation.text or "(the agent returned nothing)",
            "summary": invocation.summary[:160],
            "tokens": invocation.total_tokens,
            "latency_ms": invocation.latency_ms,
            "cost_usd": float(invocation.new_tokens),
        }
        if invocation.claimed_hypothesis is not None:
            report["claimed_hypothesis"] = invocation.claimed_hypothesis
        return report


def _prompt_for(brief: dict) -> str:
    context = brief["context"]
    ranked = "\n".join(
        f"  [{h['index']}] {h['label']}  p={h['probability']:.3f}"
        for h in context["belief_ranked"]
    )
    return (
        f"TASK\n{context['task']}\n\n"
        f"STEP {context['step']}\n\n"
        f"COMPETING HYPOTHESES (current belief)\n{ranked}\n\n"
        f"WHERE THE TEAM STANDS\n"
        f"  entropy {context['entropy_bits']} bits, confidence {context['confidence']}\n"
        f"  quality {context['quality']}, verification {context['verification_score']}\n"
        f"  {context['unresolved_subtasks']} subtasks unresolved\n\n"
        f"YOUR BRIEF\n{brief['instruction']}\n"
        f"{REPORT_CONTRACT}"
    )


def _parse_verdict(text: str) -> tuple[dict, bool]:
    """Pull the json block out of the reply, and degrade honestly when it is absent."""
    match = JSON_BLOCK.search(text) or BARE_OBJECT.search(text)
    if match:
        try:
            raw = json.loads(match.group(1))
        except json.JSONDecodeError:
            raw = None
        if isinstance(raw, dict):
            outcome = str(raw.get("outcome", "partial")).strip().lower()
            claimed = raw.get("claimed_hypothesis")
            return (
                {
                    "outcome": outcome if outcome in {"success", "partial", "failure"} else "partial",
                    "confidence": _clamp(raw.get("confidence", 0.5)),
                    "summary": str(raw.get("summary") or "").strip(),
                    "claimed_hypothesis": int(claimed) if isinstance(claimed, (int, float)) else None,
                },
                True,
            )

    # An agent that would not state a verdict has not made a claim, so it concentrates no belief
    # mass. Calling that a success would invent a result the reply does not contain.
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return (
        {
            "outcome": "partial",
            "confidence": 0.2,
            "summary": first_line[:160] or "no structured verdict returned",
            "claimed_hypothesis": None,
        },
        False,
    )


def _without_json_block(text: str) -> str:
    return JSON_BLOCK.sub("", text).strip()


def _clamp(value: object) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
