"use client";

import { useGame } from "@/lib/store";

function Meter({
  label,
  value,
  display,
  color,
}: {
  label: string;
  value: number;
  display: string;
  color: string;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="stat-label">{label}</span>
        <span className="font-mono text-2xs tabular-nums" style={{ color }}>
          {display}
        </span>
      </div>
      <div className="meter mt-1">
        <div
          className="meter-fill"
          style={{ width: `${Math.max(0, Math.min(100, value * 100))}%`, color }}
        />
      </div>
    </div>
  );
}

function Stat({ label, value, tone = "#7bf7c4" }: { label: string; value: string; tone?: string }) {
  return (
    <div className="border-2 border-edge bg-ink px-2 py-1">
      <p className="stat-label">{label}</p>
      <p className="font-mono text-xs tabular-nums" style={{ color: tone }}>
        {value}
      </p>
    </div>
  );
}

export function HUD() {
  const { run, lastStep } = useGame();
  const state = run?.state;
  if (!state) return null;

  const gain = lastStep?.information_gain ?? 0;

  return (
    <div className="panel space-y-3 px-4 py-3">
      <div className="flex items-baseline justify-between">
        <h3 className="font-pixel text-2xs text-phosphor">STATUS</h3>
        <span className="font-pixel text-3xs text-amber">
          SCORE {run.cumulative_reward >= 0 ? "" : "-"}
          {Math.abs(run.cumulative_reward).toFixed(2)}
        </span>
      </div>

      <Meter
        label="Fuel (budget)"
        value={state.budget_remaining}
        display={`$${state.budget_remaining_usd.toFixed(3)}`}
        color="#ffc857"
      />
      <Meter
        label="Time"
        value={state.latency_remaining}
        display={`${(state.latency_consumed_ms / 1000).toFixed(1)}s`}
        color="#5fe3ff"
      />
      <Meter
        label="Confidence"
        value={state.confidence}
        display={state.confidence.toFixed(3)}
        color="#7bf7c4"
      />
      <Meter
        label="Quality"
        value={state.quality}
        display={state.quality.toFixed(3)}
        color="#ff5fd2"
      />
      <Meter
        label="Verified"
        value={state.verification_score}
        display={state.verification_score.toFixed(3)}
        color="#b8ff5f"
      />

      <div className="grid grid-cols-2 gap-2 pt-1">
        <Stat label="Entropy" value={`${state.entropy.toFixed(3)} bits`} tone="#5fe3ff" />
        <Stat
          label="Info gain"
          value={`${gain >= 0 ? "+" : ""}${gain.toFixed(3)}`}
          tone={gain >= 0 ? "#b8ff5f" : "#ff5f6d"}
        />
        <Stat label="Step" value={String(state.step)} tone="#ffc857" />
        <Stat
          label="Subtasks"
          value={`${state.unresolved_subtasks}/${state.total_subtasks}`}
          tone="#a78bfa"
        />
        <Stat label="Tokens" value={state.tokens_consumed.toLocaleString()} tone="#a78bfa" />
        <Stat label="Spent" value={`$${state.budget_spent_usd.toFixed(3)}`} tone="#ffc857" />
      </div>

      {/* The entropy calculation, shown rather than asserted. */}
      {lastStep ? (
        <div className="border-2 border-edge bg-ink px-2 py-2">
          <p className="stat-label">Information gain</p>
          <p className="mt-1 font-mono text-2xs leading-relaxed text-mist">
            H(before) {lastStep.entropy_before.toFixed(4)} − H(after){" "}
            {lastStep.entropy_after.toFixed(4)} ={" "}
            <span className={gain >= 0 ? "text-lime" : "text-crimson"}>
              {gain >= 0 ? "+" : ""}
              {gain.toFixed(4)} bits
            </span>
          </p>
        </div>
      ) : null}

      {run.terminated ? (
        <div className="border-2 border-crimson bg-ink px-2 py-2 text-center">
          <p className="font-pixel text-2xs text-crimson">GAME OVER</p>
          <p className="mt-1 font-mono text-3xs text-[#ffb3b8]">
            {run.termination_reason?.replace(/_/g, " ")}
          </p>
        </div>
      ) : null}
    </div>
  );
}
