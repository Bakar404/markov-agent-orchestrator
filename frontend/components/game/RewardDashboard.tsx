"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { PixelSprite } from "@/components/pixel/PixelSprite";
import { useGame } from "@/lib/store";

const AXIS = {
  stroke: "#8b84c9",
  tick: { fill: "#e6e3ff", fontSize: 12, fontFamily: "IBM Plex Mono, monospace" },
};

const GRID = "#332f60";

const TOOLTIP_STYLE = {
  backgroundColor: "#0b0a14",
  border: "2px solid #8b84c9",
  borderRadius: 0,
  color: "#e6e3ff",
  fontFamily: "IBM Plex Mono, monospace",
  fontSize: 13,
};

const TERM_COLORS: Record<string, string> = {
  quality: "#ff5fd2",
  verification: "#b8ff5f",
  information_gain: "#5fe3ff",
  progress: "#a78bfa",
  cost: "#ffc857",
  latency: "#fb923c",
  duplicate: "#ff5f6d",
  terminal: "#7bf7c4",
};

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel px-4 py-3">
      <h3 className="font-pixel text-2xs text-phosphor">{title}</h3>
      <div className="mt-3">{children}</div>
    </section>
  );
}

export function RewardDashboard() {
  const { metrics } = useGame();

  if (!metrics || metrics.series.length === 0) {
    return (
      <div className="panel px-4 py-6 text-center">
        <p className="font-mono text-xs text-edge">
          No steps recorded yet. Run the simulation, then hit ⟳ Stats.
        </p>
      </div>
    );
  }

  const { totals, series, per_agent: perAgent, reward_terms: rewardTerms } = metrics;

  const termData = Object.entries(rewardTerms)
    .map(([term, value]) => ({ term, value }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        {[
          { label: "Cumulative reward", value: totals.cumulative_reward.toFixed(3), tone: "#7bf7c4" },
          {
            label: "Cost efficiency",
            value: `${totals.cost_efficiency.toFixed(2)} R/$`,
            tone: "#ffc857",
          },
          {
            label: "Total info gain",
            value: `${totals.total_information_gain.toFixed(3)} bits`,
            tone: "#5fe3ff",
          },
          { label: "Quality score", value: totals.quality.toFixed(3), tone: "#ff5fd2" },
        ].map((stat) => (
          <div key={stat.label} className="panel-tight px-3 py-2">
            <p className="stat-label">{stat.label}</p>
            <p className="font-mono text-sm tabular-nums" style={{ color: stat.tone }}>
              {stat.value}
            </p>
          </div>
        ))}
      </div>

      <Panel title="CUMULATIVE REWARD">
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={series} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 2" />
            <XAxis dataKey="step" {...AXIS} />
            <YAxis {...AXIS} />
            <Tooltip contentStyle={TOOLTIP_STYLE} itemStyle={{ color: "#e6e3ff" }} />
            <ReferenceLine y={0} stroke="#8b84c9" />
            <Line
              type="stepAfter"
              dataKey="cumulative_reward"
              stroke="#7bf7c4"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="stepAfter"
              dataKey="reward"
              stroke="#ff5fd2"
              strokeWidth={1}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
        <p className="font-mono text-3xs text-edge">
          <span className="text-phosphor">━</span> cumulative ·{" "}
          <span className="text-magenta">━</span> per step
        </p>
      </Panel>

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel title="ENTROPY &amp; INFORMATION GAIN">
          <ResponsiveContainer width="100%" height={170}>
            <AreaChart data={series} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="2 2" />
              <XAxis dataKey="step" {...AXIS} />
              <YAxis {...AXIS} />
              <Tooltip contentStyle={TOOLTIP_STYLE} itemStyle={{ color: "#e6e3ff" }} />
              <ReferenceLine y={0} stroke="#8b84c9" />
              <Area
                type="monotone"
                dataKey="entropy_after"
                stroke="#5fe3ff"
                fill="#5fe3ff"
                fillOpacity={0.16}
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey="information_gain"
                stroke="#b8ff5f"
                fill="#b8ff5f"
                fillOpacity={0.24}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
          <p className="font-mono text-3xs text-edge">
            <span className="text-cyan">━</span> H(s) bits ·{" "}
            <span className="text-lime">━</span> gain per step (negative = ambiguity found)
          </p>
        </Panel>

        <Panel title="REWARD DECOMPOSITION">
          <ResponsiveContainer width="100%" height={170}>
            <BarChart data={termData} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="2 2" />
              <XAxis dataKey="term" {...AXIS} tickFormatter={(t: string) => t.slice(0, 4)} />
              <YAxis {...AXIS} />
              <Tooltip contentStyle={TOOLTIP_STYLE} itemStyle={{ color: "#e6e3ff" }} />
              <ReferenceLine y={0} stroke="#8b84c9" />
              <Bar dataKey="value" isAnimationActive={false}>
                {termData.map((entry) => (
                  <Cell key={entry.term} fill={TERM_COLORS[entry.term] ?? "#a78bfa"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="font-mono text-3xs text-edge">
            Summed across every step. Negative bars are the cost of doing business.
          </p>
        </Panel>
      </div>

      <Panel title="REWARD BY AGENT">
        <div className="space-y-2">
          {perAgent
            .slice()
            .sort((a, b) => b.reward - a.reward)
            .map((agent) => {
              const peak = Math.max(...perAgent.map((a) => Math.abs(a.reward)), 0.001);
              const width = (Math.abs(agent.reward) / peak) * 100;
              return (
                <div key={agent.agent_id} className="flex items-center gap-3">
                  <PixelSprite id={agent.agent_id} size={32} />
                  <div className="w-24 shrink-0">
                    <p className="font-pixel text-3xs" style={{ color: agent.color }}>
                      {agent.label.replace(" Agent", "")}
                    </p>
                    <p className="font-mono text-3xs text-edge">
                      x{agent.invocations} · ${agent.cost.toFixed(3)}
                    </p>
                  </div>
                  <div className="meter flex-1">
                    <div
                      className="meter-fill"
                      style={{
                        width: `${width}%`,
                        color: agent.reward >= 0 ? agent.color : "#ff5f6d",
                      }}
                    />
                  </div>
                  <span
                    className="w-16 text-right font-mono text-3xs tabular-nums"
                    style={{ color: agent.reward >= 0 ? "#b8ff5f" : "#ff5f6d" }}
                  >
                    {agent.reward >= 0 ? "+" : ""}
                    {agent.reward.toFixed(3)}
                  </span>
                  <span className="w-20 text-right font-mono text-3xs text-amber">
                    {agent.cost_efficiency.toFixed(1)} R/$
                  </span>
                </div>
              );
            })}
        </div>
      </Panel>
    </div>
  );
}
