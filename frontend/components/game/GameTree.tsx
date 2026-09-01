"use client";

import { useMemo } from "react";

import { useGame } from "@/lib/store";
import type { Trace } from "@/lib/types";

const OUTCOME_TONE: Record<string, string> = {
  success: "text-lime",
  partial: "text-amber",
  failure: "text-crimson",
  terminal: "text-cyan",
  noop: "text-muted",
};

/** Actions that are legal before the escalation gate opens. Everything else is off the tree. */
const PRE_ESCALATION = new Set(["invoke_generalist", "escalate", "terminate"]);

function bits(value: number): string {
  return `${value.toFixed(2)} bits`;
}

/**
 * The orchestrator's move at one information set: the mixed strategy it drew from, with the
 * branch it actually took marked. Every branch here is a real alternative the policy scored.
 */
function DecisionNode({ trace, escalated }: { trace: Trace; escalated: boolean }) {
  const branches = Object.entries(trace.action_distribution)
    .filter(([, p]) => p > 0.0005)
    .sort((a, b) => b[1] - a[1]);

  return (
    <div className="mt-1">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="font-pixel text-3xs text-phosphor">● DECISION</span>
        <span className="font-mono text-3xs text-muted">
          orchestrator · σ(a|s) over {branches.length} legal{" "}
          {branches.length === 1 ? "action" : "actions"}
        </span>
        {!escalated ? (
          <span className="border border-amber px-1 font-mono text-3xs text-amber">
            gate shut — specialists illegal
          </span>
        ) : null}
      </div>

      <ul className="mt-1 space-y-0.5 border-l-2 border-edge pl-3">
        {branches.map(([action, probability]) => {
          const chosen = action === trace.action;
          const gated = !escalated && !PRE_ESCALATION.has(action);
          return (
            <li key={action} className="flex items-center gap-2">
              <span className="text-edge">{chosen ? "├▶" : "├─"}</span>
              <span
                className={`w-40 shrink-0 font-mono text-3xs ${
                  chosen ? "text-phosphor" : gated ? "text-edge" : "text-muted"
                }`}
              >
                {action}
              </span>
              <span className="h-1.5 w-24 shrink-0 border border-edge">
                <span
                  className={`block h-full ${chosen ? "bg-phosphor" : "bg-edge"}`}
                  style={{ width: `${Math.max(2, probability * 100)}%` }}
                />
              </span>
              <span
                className={`font-mono text-3xs tabular-nums ${
                  chosen ? "text-phosphor" : "text-muted"
                }`}
              >
                {probability.toFixed(3)}
              </span>
              {chosen ? <span className="font-pixel text-3xs text-phosphor">chosen</span> : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/**
 * Nature's move. Only the realised branch was sampled, so the alternatives are shown as
 * residual mass rather than invented: the engine never enumerated them.
 */
function ChanceNode({ trace }: { trace: Trace }) {
  const residual = Math.max(0, 1 - trace.transition_probability);

  return (
    <div className="mt-1">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="font-pixel text-3xs text-magenta">○ CHANCE</span>
        <span className="font-mono text-3xs text-muted">
          nature · P(o|s,a) = {trace.transition_probability.toFixed(3)}
        </span>
      </div>

      <ul className="mt-1 space-y-0.5 border-l-2 border-edge pl-3">
        <li className="flex flex-wrap items-center gap-2">
          <span className="text-edge">├▶</span>
          <span
            className={`w-40 shrink-0 font-mono text-3xs ${
              OUTCOME_TONE[trace.outcome] ?? "text-muted"
            }`}
          >
            {trace.outcome}
          </span>
          <span className="font-mono text-3xs tabular-nums text-muted">
            ΔH {trace.information_gain >= 0 ? "−" : "+"}
            {Math.abs(trace.information_gain).toFixed(2)} bits
          </span>
          <span
            className={`font-mono text-3xs tabular-nums ${
              trace.reward >= 0 ? "text-lime" : "text-crimson"
            }`}
          >
            r = {trace.reward >= 0 ? "+" : ""}
            {trace.reward.toFixed(3)}
          </span>
          <span className="font-mono text-3xs tabular-nums text-muted">
            ${trace.cost_usd.toFixed(4)}
          </span>
        </li>
        {residual > 0.0005 ? (
          <li className="flex items-center gap-2">
            <span className="text-edge">└─</span>
            <span className="w-40 shrink-0 font-mono text-3xs text-edge">other outcomes</span>
            <span className="font-mono text-3xs tabular-nums text-edge">
              {residual.toFixed(3)} — not sampled
            </span>
          </li>
        ) : null}
      </ul>
    </div>
  );
}

function Ply({ trace, escalated }: { trace: Trace; escalated: boolean }) {
  const coalition = trace.agents.length > 1;

  return (
    <li className="border-2 border-edge bg-ink px-3 py-2">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="font-pixel text-3xs text-amber">
          h{String(trace.step).padStart(2, "0")}
        </span>
        <span className="font-mono text-3xs text-cyan">
          info set · H(b) = {bits(trace.entropy_before)}
        </span>
        {coalition ? (
          <span className="border border-violet px-1 font-mono text-3xs text-violet">
            coalition {"{"}
            {trace.agents.join(", ")}
            {"}"}
          </span>
        ) : trace.agents.length === 1 ? (
          <span className="font-mono text-3xs text-muted">player: {trace.agents[0]}</span>
        ) : null}
      </div>

      <DecisionNode trace={trace} escalated={escalated} />
      <ChanceNode trace={trace} />
    </li>
  );
}

export function GameTree() {
  const { traces, run } = useGame();

  // The gate opens once and stays open, so escalation status is a prefix scan over the trace.
  const escalationStep = useMemo(() => {
    const hit = traces.find((t) => t.action === "escalate");
    return hit ? hit.step : null;
  }, [traces]);

  const terminal = traces.length > 0 ? traces[traces.length - 1] : null;

  if (traces.length === 0) {
    return (
      <section className="panel px-4 py-8 text-center">
        <p className="font-mono text-xs text-muted">
          No plies yet. Step the run to build the game tree.
        </p>
      </section>
    );
  }

  return (
    <section className="space-y-3">
      <header className="panel px-4 py-3">
        <h2 className="font-pixel text-2xs text-phosphor">EXTENSIVE FORM</h2>
        <p className="mt-2 max-w-4xl font-mono text-3xs leading-relaxed text-muted">
          The run as a game tree. Each ply is an orchestrator move at an information set,
          followed by a move by nature. The orchestrator never observes which hypothesis is
          true — it holds a belief, and <span className="text-cyan">H(b)</span> is how much it
          still does not know. Branch probabilities are the policy&apos;s own mixed strategy,
          read from the persisted trace.
        </p>
        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 font-mono text-3xs text-muted">
          <span>
            <span className="text-phosphor">●</span> orchestrator decision
          </span>
          <span>
            <span className="text-magenta">○</span> chance / nature
          </span>
          <span>
            <span className="text-cyan">H(b)</span> information-set entropy
          </span>
          <span>
            <span className="text-violet">{"{…}"}</span> coalition move
          </span>
        </div>
        {escalationStep !== null ? (
          <p className="mt-2 font-mono text-3xs text-amber">
            Gate opened at ply h{String(escalationStep).padStart(2, "0")} — the action set expands
            from 3 legal moves to the full specialist roster and coalitions over it.
          </p>
        ) : (
          <p className="mt-2 font-mono text-3xs text-amber">
            Gate never opened — the tree stays on its solo branch for the whole episode.
          </p>
        )}
      </header>

      <ol className="space-y-2">
        {traces.map((trace) => (
          <Ply
            key={trace.id}
            trace={trace}
            escalated={escalationStep !== null && trace.step > escalationStep}
          />
        ))}
      </ol>

      {terminal ? (
        <div className="panel px-4 py-3">
          <div className="flex flex-wrap items-baseline gap-2">
            <span className="font-pixel text-3xs text-cyan">■ TERMINAL</span>
            <span className="font-mono text-3xs text-muted">
              payoff after {traces.length} {traces.length === 1 ? "ply" : "plies"}
            </span>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1 font-mono text-3xs tabular-nums text-muted">
            <span>
              Σr ={" "}
              <span className={terminal.cumulative_reward >= 0 ? "text-lime" : "text-crimson"}>
                {terminal.cumulative_reward.toFixed(3)}
              </span>
            </span>
            <span>
              residual H(b) = <span className="text-cyan">{bits(terminal.entropy_after)}</span>
            </span>
            <span>
              confidence = <span className="text-phosphor">{terminal.confidence.toFixed(3)}</span>
            </span>
            {run?.total_cost !== undefined ? (
              <span>
                cost = <span className="text-amber">${run.total_cost.toFixed(4)}</span>
              </span>
            ) : null}
          </div>
          <p className="mt-2 font-mono text-3xs text-edge">
            Internal reward ranks moves inside one episode. It is deliberately not used to rank
            arms against each other — see /compare.
          </p>
        </div>
      ) : null}
    </section>
  );
}
