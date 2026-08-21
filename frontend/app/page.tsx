"use client";

import Link from "next/link";
import { useState } from "react";

import { Arena } from "@/components/game/Arena";
import { Controls } from "@/components/game/Controls";
import { GraphView } from "@/components/game/GraphView";
import { HUD } from "@/components/game/HUD";
import { RewardDashboard } from "@/components/game/RewardDashboard";
import { TitleScreen } from "@/components/game/TitleScreen";
import { TraceExplorer } from "@/components/game/TraceExplorer";
import { useGame } from "@/lib/store";

const TABS = [
  { id: "arena", label: "ARENA" },
  { id: "graph", label: "GRAPH" },
  { id: "rewards", label: "REWARDS" },
  { id: "traces", label: "TRACES" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function Page() {
  const { phase, run, feed, error, refreshAnalytics } = useGame();
  const [tab, setTab] = useState<TabId>("arena");

  if (phase !== "arena" || !run) {
    return <TitleScreen />;
  }

  return (
    <main className="mx-auto max-w-[110rem] space-y-3 p-3">
      <header className="panel flex flex-wrap items-baseline justify-between gap-2 px-4 py-2">
        <div>
          <h1 className="font-pixel text-xs text-phosphor">MARKOV ORCHESTRATOR</h1>
          <p className="mt-1 max-w-3xl font-mono text-3xs text-[#8f89c9]">{run.task}</p>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/research" className="font-pixel text-3xs text-violet hover:text-phosphor">
            RESEARCH ▸
          </Link>
          <div className="text-right">
            <p className="font-pixel text-3xs text-amber">{run.policy.toUpperCase()}</p>
            <p className="font-mono text-3xs text-edge">SEED {run.seed}</p>
          </div>
        </div>
      </header>

      {error ? (
        <div className="panel border-crimson px-4 py-2">
          <p className="font-mono text-xs text-[#ffb3b8]">{error}</p>
        </div>
      ) : null}

      <Controls />

      <nav className="flex gap-1">
        {TABS.map((entry) => (
          <button
            key={entry.id}
            type="button"
            onClick={() => {
              setTab(entry.id);
              if (entry.id !== "arena") void refreshAnalytics();
            }}
            className={`border-2 px-4 py-2 font-pixel text-3xs ${
              tab === entry.id
                ? "border-phosphor bg-phosphor text-void"
                : "border-edge bg-slab text-edge hover:text-phosphor"
            }`}
          >
            {entry.label}
          </button>
        ))}
      </nav>

      {tab === "arena" ? (
        <div className="grid gap-3 lg:grid-cols-[1fr_20rem]">
          <Arena />
          <HUD />
        </div>
      ) : null}

      {tab === "graph" ? (
        <div className="grid gap-3 lg:grid-cols-[1fr_20rem]">
          <GraphView />
          <HUD />
        </div>
      ) : null}

      {tab === "rewards" ? <RewardDashboard /> : null}

      {tab === "traces" ? <TraceExplorer /> : null}

      {tab === "arena" ? (
        <section className="panel px-4 py-3">
        <h2 className="font-pixel text-2xs text-phosphor">EVENT LOG</h2>
        <ul className="mt-2 max-h-56 space-y-1 overflow-y-auto">
          {feed.length === 0 ? (
            <li className="font-mono text-3xs text-edge">
              No steps yet. Press Start or Step to begin.
            </li>
          ) : null}
          {feed.map((step) => (
            <li key={step.step} className="font-mono text-3xs leading-relaxed">
              <span className="text-amber">[{String(step.step).padStart(3, "0")}]</span>{" "}
              <span
                className={
                  step.outcome === "success"
                    ? "text-lime"
                    : step.outcome === "failure"
                      ? "text-crimson"
                      : "text-amber"
                }
              >
                {step.outcome.toUpperCase().padEnd(7)}
              </span>{" "}
              <span className="text-[#c9c4ff]">{step.notes}</span>
            </li>
          ))}
        </ul>
      </section>
      ) : null}
    </main>
  );
}
