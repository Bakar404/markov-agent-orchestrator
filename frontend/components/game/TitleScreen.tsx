"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { PixelSprite } from "@/components/pixel/PixelSprite";
import { useGame } from "@/lib/store";

const ROSTER = ["planner", "researcher", "critic", "verifier", "memory", "executor"];

const SAMPLE_TASKS = [
  "Which caching strategy fits a read-heavy API with bursty traffic",
  "Should we shard by tenant or by entity for a write-heavy multi-tenant store",
  "Which exploration strategy suits a non-stationary orchestration MDP",
  "Plan a migration from REST to gRPC for a service with 40 consumers",
];

const ESCALATION_TONE: Record<string, string> = {
  never: "text-amber",
  always: "text-crimson",
  heuristic: "text-cyan",
  learned: "text-violet",
};

export function TitleScreen() {
  const { meta, loadMeta, startGame, booting, error, clearError } = useGame();
  const [screen, setScreen] = useState<"attract" | "setup">("attract");

  const [task, setTask] = useState(SAMPLE_TASKS[0]);
  const [strategy, setStrategy] = useState("cascade");
  const [complexity, setComplexity] = useState(0.55);
  const [budget, setBudget] = useState(1.2);
  const [seed, setSeed] = useState<string>("");
  const [experiment, setExperiment] = useState("");

  useEffect(() => {
    void loadMeta();
  }, [loadMeta]);

  useEffect(() => {
    if (screen !== "attract") return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        setScreen("setup");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [screen]);

  const offline = Boolean(error);

  if (screen === "attract") {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center gap-10 px-6">
        <div className="text-center">
          <h1
            className="font-pixel text-3xl leading-tight text-phosphor sm:text-5xl"
            style={{ textShadow: "4px 4px 0 #07060f, 8px 8px 0 rgba(255,95,210,0.45)" }}
          >
            MARKOV
          </h1>
          <p
            className="mt-4 font-pixel text-xs text-magenta sm:text-base"
            style={{ textShadow: "3px 3px 0 #07060f" }}
          >
            AGENT ORCHESTRATOR
          </p>
        </div>

        <div className="flex items-end gap-4 sm:gap-7">
          {ROSTER.map((id, index) => (
            <div
              key={id}
              className="animate-bob"
              style={{ animationDelay: `${index * 0.16}s` }}
            >
              <PixelSprite id={id} size={72} animation="idle" />
            </div>
          ))}
        </div>

        <div className="panel max-w-2xl px-6 py-4 text-center">
          <p className="font-mono text-xs leading-relaxed text-[#a9a3e0]">
            One agent starts. Six more wait behind an escalation gate that costs budget to open.
            The question is whether opening it was worth it — so every strategy is measured
            against a single agent doing the same task alone.
          </p>
        </div>

        <div className="panel max-w-2xl px-6 py-4">
          <p className="stat-label text-center">Run a real experiment</p>
          <p className="mt-2 text-center font-mono text-3xs leading-relaxed text-[#8f89c9]">
            Your agent does the work; the arena decides who acts and records what it cost.
          </p>
          <pre className="mt-3 overflow-x-auto border-2 border-edge bg-ink px-3 py-2 font-mono text-3xs text-phosphor">
{`cd ${"C:\\src\\markov-agent-orchestrator"}
copilot

> compare orchestration against a single agent on: <your task>`}
          </pre>
          <p className="mt-2 text-center font-mono text-3xs text-edge">
            Results land in{" "}
            <Link href="/compare" className="text-cyan hover:text-phosphor">
              /compare
            </Link>
            . Live steps cost real credits.
          </p>
        </div>

        <button
          type="button"
          onClick={() => setScreen("setup")}
          className="font-pixel text-3xs text-amber hover:text-phosphor sm:text-2xs"
        >
          ▶ OR EXPLORE IN SIMULATION (FREE)
        </button>

        {offline ? (
          <p className="font-pixel text-3xs text-crimson">
            BACKEND OFFLINE — START THE API ON PORT 8000
          </p>
        ) : (
          <p className="font-pixel text-3xs text-edge">INSERT COIN · V0.1 · 2026</p>
        )}

        <nav className="flex items-center gap-5">
          <Link href="/compare" className="font-pixel text-3xs text-amber hover:text-phosphor">
            COMPARE ▸
          </Link>
          <Link href="/campaign" className="font-pixel text-3xs text-cyan hover:text-phosphor">
            LEARNING LAB ▸
          </Link>
          <Link href="/research" className="font-pixel text-3xs text-violet hover:text-phosphor">
            STRATEGIES ▸
          </Link>
        </nav>
      </main>
    );
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-4xl flex-col justify-center gap-6 px-6 py-12">
      <header className="flex items-baseline justify-between">
        <div>
          <h2 className="font-pixel text-lg text-phosphor">SIMULATION SETUP</h2>
          <p className="mt-1 font-mono text-3xs text-[#8f89c9]">
            Outcomes are sampled, not real. Free and reproducible from the seed — use it to build
            intuition, then confirm with a live run from the CLI.
          </p>
        </div>
        <button
          type="button"
          className="font-pixel text-3xs text-edge hover:text-phosphor"
          onClick={() => setScreen("attract")}
        >
          ◀ BACK
        </button>
      </header>

      {error ? (
        <div className="panel border-crimson px-4 py-3">
          <p className="font-pixel text-3xs text-crimson">CONNECTION ERROR</p>
          <p className="mt-2 font-mono text-xs text-[#ffb3b8]">{error}</p>
          <button
            type="button"
            className="pixel-btn mt-3"
            onClick={() => {
              clearError();
              void loadMeta();
            }}
          >
            Retry
          </button>
        </div>
      ) : null}

      <section className="panel px-5 py-4">
        <label className="stat-label" htmlFor="task">
          Objective
        </label>
        <textarea
          id="task"
          value={task}
          onChange={(event) => setTask(event.target.value)}
          rows={2}
          className="mt-2 w-full resize-none border-2 border-edge bg-ink px-3 py-2 font-mono text-sm
                     text-phosphor outline-none focus:border-phosphor"
        />
        <div className="mt-2 flex flex-wrap gap-2">
          {SAMPLE_TASKS.map((sample) => (
            <button
              key={sample}
              type="button"
              onClick={() => setTask(sample)}
              className="border border-edge px-2 py-1 font-mono text-3xs text-[#8f89c9] hover:border-phosphor hover:text-phosphor"
            >
              {sample.slice(0, 38)}…
            </button>
          ))}
        </div>
      </section>

      <section className="panel px-5 py-4">
        <p className="stat-label">Strategy</p>
        <p className="mt-1 font-mono text-3xs text-[#8f89c9]">
          Every run starts with one agent. The strategy decides whether it ever escalates into
          orchestration — and that decision is what gets measured.
        </p>
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          {(meta?.strategies ?? []).map((option) => {
            const selected = option.id === strategy;
            return (
              <button
                key={option.id}
                type="button"
                onClick={() => setStrategy(option.id)}
                className={`border-2 px-3 py-2 text-left transition-none ${
                  selected
                    ? "border-phosphor bg-phosphor/10 shadow-pixel-sm"
                    : "border-edge bg-ink hover:border-violet"
                }`}
              >
                <span className="flex items-center justify-between gap-2">
                  <span className="font-pixel text-3xs text-phosphor">{option.label}</span>
                  <span
                    className={`font-pixel text-3xs ${ESCALATION_TONE[option.escalates] ?? "text-edge"}`}
                  >
                    {option.escalates.toUpperCase()}
                  </span>
                </span>
                <span className="mt-1 block font-mono text-3xs leading-relaxed text-[#8f89c9]">
                  {option.summary}
                </span>
                <span className="mt-1 block font-mono text-3xs leading-relaxed text-edge">
                  {option.when}
                </span>
              </button>
            );
          })}
          {!meta ? (
            <p className="font-mono text-xs text-edge">Loading strategies…</p>
          ) : null}
        </div>
      </section>

      <section className="panel grid gap-4 px-5 py-4 sm:grid-cols-3">
        <div>
          <label className="stat-label" htmlFor="complexity">
            Difficulty {complexity.toFixed(2)}
          </label>
          <input
            id="complexity"
            type="range"
            min={0.05}
            max={0.95}
            step={0.05}
            value={complexity}
            onChange={(event) => setComplexity(Number(event.target.value))}
            className="mt-2 w-full accent-magenta"
          />
        </div>
        <div>
          <label className="stat-label" htmlFor="budget">
            Fuel ${budget.toFixed(2)}
          </label>
          <input
            id="budget"
            type="range"
            min={0.2}
            max={5}
            step={0.1}
            value={budget}
            onChange={(event) => setBudget(Number(event.target.value))}
            className="mt-2 w-full accent-amber"
          />
        </div>
        <div>
          <label className="stat-label" htmlFor="seed">
            Seed (blank = random)
          </label>
          <input
            id="seed"
            value={seed}
            onChange={(event) => setSeed(event.target.value.replace(/[^0-9]/g, ""))}
            placeholder="random"
            className="mt-2 w-full border-2 border-edge bg-ink px-3 py-2 font-mono text-sm
                       text-phosphor outline-none focus:border-phosphor"
          />
        </div>
      </section>

      <section className="panel px-5 py-4">
        <label className="stat-label" htmlFor="experiment">
          Experiment (optional)
        </label>
        <input
          id="experiment"
          value={experiment}
          onChange={(event) => setExperiment(event.target.value)}
          placeholder="worth-it"
          className="mt-2 w-full border-2 border-edge bg-ink px-3 py-2 font-mono text-sm
                     text-phosphor outline-none focus:border-phosphor"
        />
        <p className="mt-2 font-mono text-3xs leading-relaxed text-[#8f89c9]">
          Name an experiment to compare strategies. Run the same task and the{" "}
          <span className="text-amber">same seed</span> once per strategy, including{" "}
          <span className="text-amber">Single Agent (control)</span> — differences are only paired
          on seeds both arms ran. Results appear under{" "}
          <Link href="/compare" className="text-cyan hover:text-phosphor">
            /compare
          </Link>
          .
        </p>
      </section>

      <button
        type="button"
        disabled={booting || task.trim().length < 3}
        onClick={() =>
          void startGame({
            task: task.trim(),
            strategy,
            seed: seed ? Number(seed) : null,
            task_complexity: complexity,
            budget_usd: budget,
            ...(experiment.trim() ? { experiment: experiment.trim() } : {}),
          })
        }
        className="pixel-btn pixel-btn-primary py-4 text-sm"
      >
        {booting ? "BOOTING…" : "▶ LAUNCH RUN"}
      </button>
    </main>
  );
}
