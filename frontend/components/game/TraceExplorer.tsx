"use client";

import { useState } from "react";

import { useGame } from "@/lib/store";
import type { Trace } from "@/lib/types";

const OUTCOME_TONE: Record<string, string> = {
  success: "text-lime",
  partial: "text-amber",
  failure: "text-crimson",
  terminal: "text-cyan",
  noop: "text-edge",
};

function Distribution({ trace }: { trace: Trace }) {
  const entries = Object.entries(trace.action_distribution)
    .filter(([, p]) => p > 0.0005)
    .sort((a, b) => b[1] - a[1]);
  const peak = Math.max(...entries.map(([, p]) => p), 0.001);

  return (
    <div className="space-y-1">
      {entries.map(([action, probability]) => {
        const chosen = action === trace.action;
        return (
          <div key={action} className="flex items-center gap-2">
            <span
              className={`w-36 shrink-0 font-mono text-3xs ${
                chosen ? "text-phosphor" : "text-[#8f89c9]"
              }`}
            >
              {chosen ? "▶ " : "  "}
              {action}
            </span>
            <div className="meter h-2 flex-1">
              <div
                className="meter-fill"
                style={{
                  width: `${(probability / peak) * 100}%`,
                  color: chosen ? "#7bf7c4" : "#6f68ab",
                }}
              />
            </div>
            <span className="w-14 text-right font-mono text-3xs tabular-nums text-[#8f89c9]">
              {probability.toFixed(4)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function Detail({ trace }: { trace: Trace }) {
  const breakdown = trace.reward_breakdown;
  const terms = Object.entries(breakdown).filter(
    ([key]) => !["per_agent", "total"].includes(key),
  ) as [string, number][];

  return (
    <div className="grid gap-4 border-t-2 border-edge bg-ink px-3 py-3 lg:grid-cols-3">
      <div>
        <p className="stat-label">Policy distribution</p>
        <div className="mt-2">
          <Distribution trace={trace} />
        </div>
      </div>

      <div>
        <p className="stat-label">Reward decomposition</p>
        <table className="mt-2 w-full font-mono text-3xs">
          <tbody>
            {terms.map(([term, value]) => (
              <tr key={term}>
                <td className="py-[1px] text-[#8f89c9]">{term}</td>
                <td
                  className={`py-[1px] text-right tabular-nums ${
                    value >= 0 ? "text-lime" : "text-crimson"
                  }`}
                >
                  {value >= 0 ? "+" : ""}
                  {value.toFixed(4)}
                </td>
              </tr>
            ))}
            <tr className="border-t border-edge">
              <td className="pt-1 text-phosphor">total</td>
              <td className="pt-1 text-right tabular-nums text-phosphor">
                {breakdown.total >= 0 ? "+" : ""}
                {breakdown.total.toFixed(4)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div>
        <p className="stat-label">State transition</p>
        <table className="mt-2 w-full font-mono text-3xs">
          <tbody>
            <tr>
              <td className="text-[#8f89c9]">from state</td>
              <td className="text-right text-edge">
                {trace.prev_state_id?.slice(0, 8) ?? "genesis"}
              </td>
            </tr>
            <tr>
              <td className="text-[#8f89c9]">to state</td>
              <td className="text-right text-edge">{trace.state_id.slice(0, 8)}</td>
            </tr>
            <tr>
              <td className="text-[#8f89c9]">p(action)</td>
              <td className="text-right text-cyan">{trace.action_probability.toFixed(4)}</td>
            </tr>
            <tr>
              <td className="text-[#8f89c9]">p(outcome)</td>
              <td className="text-right text-magenta">
                {trace.transition_probability.toFixed(4)}
              </td>
            </tr>
            <tr>
              <td className="text-[#8f89c9]">H before</td>
              <td className="text-right text-edge">{trace.entropy_before.toFixed(4)}</td>
            </tr>
            <tr>
              <td className="text-[#8f89c9]">H after</td>
              <td className="text-right text-edge">{trace.entropy_after.toFixed(4)}</td>
            </tr>
            <tr>
              <td className="text-[#8f89c9]">info gain</td>
              <td
                className={`text-right ${
                  trace.information_gain >= 0 ? "text-lime" : "text-crimson"
                }`}
              >
                {trace.information_gain >= 0 ? "+" : ""}
                {trace.information_gain.toFixed(4)} bits
              </td>
            </tr>
            <tr>
              <td className="text-[#8f89c9]">tokens</td>
              <td className="text-right text-violet">{trace.tokens.toLocaleString()}</td>
            </tr>
          </tbody>
        </table>
        <p className="mt-2 font-mono text-3xs leading-relaxed text-edge">{trace.notes}</p>
      </div>
    </div>
  );
}

export function TraceExplorer() {
  const { traces } = useGame();
  const [expanded, setExpanded] = useState<string | null>(null);

  if (traces.length === 0) {
    return (
      <div className="panel px-4 py-6 text-center">
        <p className="font-mono text-xs text-edge">
          No traces yet. Every orchestration step writes one; run the simulation then hit ⟳ Stats.
        </p>
      </div>
    );
  }

  return (
    <section className="panel overflow-hidden">
      <div className="flex items-baseline justify-between px-4 py-3">
        <h3 className="font-pixel text-2xs text-phosphor">EXECUTION TRACES</h3>
        <span className="font-mono text-3xs text-edge">{traces.length} steps recorded</span>
      </div>

      <div className="max-h-[32rem] overflow-y-auto">
        <table className="w-full border-collapse font-mono text-3xs">
          <thead className="sticky top-0 bg-slabLight">
            <tr className="text-left">
              {["#", "action", "agents", "outcome", "p(a)", "ΔH", "reward", "cost", "ms"].map(
                (header) => (
                  <th key={header} className="border-y-2 border-edge px-2 py-1 stat-label">
                    {header}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {traces
              .slice()
              .reverse()
              .map((trace) => {
                const open = expanded === trace.id;
                return (
                  <>
                    <tr
                      key={trace.id}
                      onClick={() => setExpanded(open ? null : trace.id)}
                      className={`cursor-pointer border-b border-[#221f42] hover:bg-slabLight ${
                        open ? "bg-slabLight" : ""
                      }`}
                    >
                      <td className="px-2 py-1 text-amber">{trace.step}</td>
                      <td className="px-2 py-1 text-[#c9c4ff]">{trace.action}</td>
                      <td className="px-2 py-1 text-violet">
                        {trace.agents.join(", ") || "—"}
                      </td>
                      <td className={`px-2 py-1 ${OUTCOME_TONE[trace.outcome] ?? "text-edge"}`}>
                        {trace.outcome}
                      </td>
                      <td className="px-2 py-1 tabular-nums text-cyan">
                        {trace.action_probability.toFixed(3)}
                      </td>
                      <td
                        className={`px-2 py-1 tabular-nums ${
                          trace.information_gain >= 0 ? "text-lime" : "text-crimson"
                        }`}
                      >
                        {trace.information_gain >= 0 ? "+" : ""}
                        {trace.information_gain.toFixed(3)}
                      </td>
                      <td
                        className={`px-2 py-1 tabular-nums ${
                          trace.reward >= 0 ? "text-lime" : "text-crimson"
                        }`}
                      >
                        {trace.reward >= 0 ? "+" : ""}
                        {trace.reward.toFixed(3)}
                      </td>
                      <td className="px-2 py-1 tabular-nums text-amber">
                        ${trace.cost_usd.toFixed(4)}
                      </td>
                      <td className="px-2 py-1 tabular-nums text-edge">
                        {trace.latency_ms.toFixed(0)}
                      </td>
                    </tr>
                    {open ? (
                      <tr key={`${trace.id}-detail`}>
                        <td colSpan={9} className="p-0">
                          <Detail trace={trace} />
                        </td>
                      </tr>
                    ) : null}
                  </>
                );
              })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
