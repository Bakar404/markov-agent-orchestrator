"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api } from "@/lib/api";
import type { CampaignPolicyResult, CampaignResponse } from "@/lib/types";

const AXIS = {
  stroke: "#8b84c9",
  tick: { fill: "#e6e3ff", fontSize: 12, fontFamily: "IBM Plex Mono, monospace" },
};
const GRID = "#332f60";
const TOOLTIP = {
  backgroundColor: "#0b0a14",
  border: "2px solid #8b84c9",
  borderRadius: 0,
  color: "#e6e3ff",
  fontFamily: "IBM Plex Mono, monospace",
  fontSize: 13,
};
const round2 = (value: number) => value.toFixed(2);

const AVAILABLE = [
  { id: "contextual_bandit", label: "Bandit", stage: 1 },
  { id: "mdp", label: "MDP", stage: 2 },
  { id: "markov_game", label: "Markov Game", stage: 3 },
  { id: "marl", label: "MARL", stage: 4 },
  { id: "heuristic", label: "Heuristic", stage: 0 },
  { id: "random", label: "Random (control)", stage: 0 },
];

/** Per-episode reward is high variance; without smoothing the curves are unreadable. */
function rollingMean(values: number[], window: number): number[] {
  return values.map((_, index) => {
    const start = Math.max(0, index - window + 1);
    const slice = values.slice(start, index + 1);
    return slice.reduce((sum, v) => sum + v, 0) / slice.length;
  });
}

function CurveChart({ result, window }: { result: CampaignPolicyResult; window: number }) {
  const data = useMemo(() => {
    const carried = result.carried.episodes.map((e) => e.reward);
    const fresh = result.fresh.episodes.map((e) => e.reward);
    const carriedSmooth = rollingMean(carried, window);
    const freshSmooth = rollingMean(fresh, window);
    return carried.map((_, i) => ({
      episode: i,
      carried: carriedSmooth[i],
      fresh: freshSmooth[i],
    }));
  }, [result, window]);

  return (
    <ResponsiveContainer width="100%" height={230}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 2" />
        <XAxis dataKey="episode" {...AXIS} />
        <YAxis {...AXIS} />
        <Tooltip
          contentStyle={TOOLTIP}
          itemStyle={{ color: "#e6e3ff" }}
          formatter={round2}
          labelFormatter={(v) => `episode ${v}`}
        />
        <Legend
          wrapperStyle={{ fontSize: 12, fontFamily: "IBM Plex Mono, monospace", color: "#e6e3ff" }}
        />
        <ReferenceLine y={0} stroke="#8b84c9" />
        <Line
          type="monotone"
          dataKey="carried"
          name="carried"
          stroke="#7bf7c4"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="fresh"
          name="fresh (control)"
          stroke="#ff5fd2"
          strokeWidth={2}
          strokeDasharray="4 3"
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function CampaignPage() {
  const [episodes, setEpisodes] = useState(40);
  const [complexity, setComplexity] = useState(0.55);
  const [budget, setBudget] = useState(1.2);
  const [seedBase, setSeedBase] = useState(1000);
  const [window, setWindow] = useState(5);
  const [selected, setSelected] = useState<string[]>([
    "contextual_bandit",
    "mdp",
    "markov_game",
    "marl",
    "random",
  ]);
  const [running, setRunning] = useState(false);
  const [data, setData] = useState<CampaignResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const toggle = (id: string) =>
    setSelected((current) =>
      current.includes(id) ? current.filter((p) => p !== id) : [...current, id],
    );

  const run = async () => {
    if (selected.length === 0) return;
    setRunning(true);
    setError(null);
    try {
      setData(
        await api.campaign({
          policies: selected,
          episodes,
          seed_base: seedBase,
          task_complexity: complexity,
          budget_usd: budget,
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  };

  const deltaData = useMemo(
    () =>
      (data?.results ?? []).map((r) => ({
        policy: r.policy.replace("contextual_", ""),
        delta: r.delta,
        // ErrorBar renders +-2 SE, matching the significance rule.
        error: 2 * r.stderr,
        significant: r.significant,
      })),
    [data],
  );

  const underpowered = episodes < 25;

  return (
    <main className="mx-auto max-w-[110rem] space-y-3 p-3">
      <header className="panel flex flex-wrap items-baseline justify-between gap-3 px-4 py-2">
        <div>
          <h1 className="font-pixel text-sm text-phosphor">CROSS-EPISODE LEARNING</h1>
          <p className="mt-2 max-w-4xl font-mono text-2xs leading-relaxed text-mist">
            Each policy runs twice over the same task instances: once carrying learned
            parameters between episodes, once fresh. The paired difference removes task
            difficulty as a confound.
          </p>
        </div>
        <Link href="/" className="font-pixel text-3xs text-violet hover:text-phosphor">
          ◂ ORCHESTRATOR
        </Link>
      </header>

      <section className="panel space-y-3 px-4 py-3">
        <div>
          <p className="stat-label">Policies</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {AVAILABLE.map((policy) => {
              const on = selected.includes(policy.id);
              return (
                <button
                  key={policy.id}
                  type="button"
                  onClick={() => toggle(policy.id)}
                  className={`border-2 px-3 py-1 font-pixel text-3xs ${
                    on
                      ? "border-phosphor bg-phosphor text-void"
                      : "border-edge bg-ink text-muted hover:text-phosphor"
                  }`}
                >
                  {policy.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-4">
          <div>
            <label className="stat-label" htmlFor="episodes">
              Episodes per arm: {episodes}
            </label>
            <input
              id="episodes"
              type="range"
              min={10}
              max={120}
              step={5}
              value={episodes}
              onChange={(e) => setEpisodes(Number(e.target.value))}
              className="mt-2 w-full accent-phosphor"
            />
          </div>
          <div>
            <label className="stat-label" htmlFor="complexity">
              Difficulty {complexity.toFixed(2)}
            </label>
            <input
              id="complexity"
              type="range"
              min={0.1}
              max={0.9}
              step={0.05}
              value={complexity}
              onChange={(e) => setComplexity(Number(e.target.value))}
              className="mt-2 w-full accent-magenta"
            />
          </div>
          <div>
            <label className="stat-label" htmlFor="budget">
              Budget ${budget.toFixed(2)}
            </label>
            <input
              id="budget"
              type="range"
              min={0.4}
              max={4}
              step={0.1}
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="mt-2 w-full accent-amber"
            />
          </div>
          <div>
            <label className="stat-label" htmlFor="smooth">
              Smoothing window: {window}
            </label>
            <input
              id="smooth"
              type="range"
              min={1}
              max={15}
              step={1}
              value={window}
              onChange={(e) => setWindow(Number(e.target.value))}
              className="mt-2 w-full accent-cyan"
            />
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            className="pixel-btn pixel-btn-primary"
            disabled={running || selected.length === 0}
            onClick={() => void run()}
          >
            {running ? "RUNNING…" : "▶ RUN EXPERIMENT"}
          </button>
          <span className="font-mono text-2xs text-muted">
            {selected.length} policies × {episodes} episodes × 2 arms ={" "}
            {selected.length * episodes * 2} runs
          </span>
          {underpowered ? (
            <span className="font-mono text-2xs text-amber">
              ⚠ under 25 episodes rarely resolves the effect above noise
            </span>
          ) : null}
        </div>
      </section>

      {error ? (
        <div className="panel border-crimson px-4 py-2">
          <p className="font-mono text-xs text-[#ffb3b8]">{error}</p>
        </div>
      ) : null}

      {data ? (
        <>
          <section className="panel overflow-x-auto px-4 py-3">
            <h2 className="font-pixel text-2xs text-phosphor">PAIRED RESULTS</h2>
            <table className="mt-3 w-full border-collapse font-mono text-2xs">
              <thead>
                <tr className="text-left">
                  {[
                    "policy",
                    "stage",
                    "carried",
                    "fresh",
                    "Δ ± SE",
                    "slope/ep",
                    "win fresh→carried",
                    "verdict",
                  ].map((h) => (
                    <th
                      key={h}
                      className="border-y-2 border-edge px-2 py-2 font-mono text-2xs uppercase tracking-wider text-muted"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.results.map((r) => (
                  <tr key={r.policy} className="border-b border-edge">
                    <td className="px-2 py-1 text-mist">{r.label}</td>
                    <td className="px-2 py-1 text-muted">{r.stage}</td>
                    <td className="px-2 py-1 tabular-nums text-phosphor">
                      {r.carried.mean_reward.toFixed(2)}
                    </td>
                    <td className="px-2 py-1 tabular-nums text-magenta">
                      {r.fresh.mean_reward.toFixed(2)}
                    </td>
                    <td
                      className={`px-2 py-1 tabular-nums ${
                        r.delta >= 0 ? "text-lime" : "text-crimson"
                      }`}
                    >
                      {r.delta >= 0 ? "+" : ""}
                      {r.delta.toFixed(2)} ± {r.stderr.toFixed(2)}
                    </td>
                    <td
                      className={`px-2 py-1 tabular-nums ${
                        r.slope >= 0 ? "text-lime" : "text-crimson"
                      }`}
                    >
                      {r.slope >= 0 ? "+" : ""}
                      {r.slope.toFixed(3)}
                    </td>
                    <td className="px-2 py-1 tabular-nums text-amber">
                      {(r.fresh.win_rate * 100).toFixed(0)}% →{" "}
                      {(r.carried.win_rate * 100).toFixed(0)}%
                    </td>
                    <td className="px-2 py-1">
                      {r.significant ? (
                        <span className={r.delta >= 0 ? "text-lime" : "text-crimson"}>
                          {r.delta >= 0 ? "✓ learns" : "✗ degrades"}
                        </span>
                      ) : (
                        <span className="text-muted">inconclusive</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 font-mono text-2xs leading-relaxed text-mist">
              {data.interpretation}
            </p>
          </section>

          <section className="panel px-4 py-3">
            <h2 className="font-pixel text-2xs text-phosphor">EFFECT SIZE</h2>
            <p className="mt-2 font-mono text-2xs leading-relaxed text-mist">
              Bars are the paired difference; whiskers are ±2 standard errors. A bar whose
              whisker clears zero is a real effect.
            </p>
            <div className="mt-3">
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={deltaData} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
                  <CartesianGrid stroke={GRID} strokeDasharray="2 2" />
                  <XAxis dataKey="policy" {...AXIS} />
                  <YAxis {...AXIS} />
                  <Tooltip contentStyle={TOOLTIP} itemStyle={{ color: "#e6e3ff" }} formatter={round2} />
                  <ReferenceLine y={0} stroke="#7bf7c4" />
                  <Bar dataKey="delta" isAnimationActive={false}>
                    {deltaData.map((entry) => (
                      <Cell
                        key={entry.policy}
                        fill={
                          !entry.significant
                            ? "#6f68ab"
                            : entry.delta >= 0
                              ? "#b8ff5f"
                              : "#ff5f6d"
                        }
                      />
                    ))}
                    <ErrorBar dataKey="error" width={6} strokeWidth={2} stroke="#ffc857" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <div className="grid gap-3 lg:grid-cols-2">
            {data.results.map((result) => (
              <section key={result.policy} className="panel px-4 py-3">
                <div className="flex items-baseline justify-between">
                  <h3 className="font-pixel text-2xs text-phosphor">
                    {result.label.toUpperCase()}
                  </h3>
                  <span className="font-mono text-2xs text-mist">
                    thirds: {result.blocks.map((b) => b.toFixed(2)).join(" → ")}
                  </span>
                </div>
                <div className="mt-2">
                  <CurveChart result={result} window={window} />
                </div>
              </section>
            ))}
          </div>
        </>
      ) : (
        <div className="panel px-4 py-8 text-center">
          <p className="font-mono text-sm text-mist">
            Configure the experiment and press Run. Expect a few seconds per hundred episodes.
          </p>
        </div>
      )}
    </main>
  );
}
