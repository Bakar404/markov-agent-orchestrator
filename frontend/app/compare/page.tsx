"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { api } from "@/lib/api";
import { SiteNav } from "@/components/game/SiteNav";
import type { ExperimentComparison, ExperimentSummary, PairedStat } from "@/lib/types";

function Stat({ stat, precision = 1 }: { stat?: PairedStat | null; precision?: number }) {
  if (!stat) return <span className="text-edge">—</span>;
  // Cheaper or better reads green; quality is inverted because more is better there.
  const tone = stat.mean_delta <= 0 ? "text-lime" : "text-crimson";
  return (
    <span className="font-mono text-3xs tabular-nums">
      <span className={tone}>
        {stat.mean_delta >= 0 ? "+" : ""}
        {stat.mean_delta.toFixed(precision)}
      </span>
      <span className="text-edge"> ±{stat.stderr.toFixed(precision)}</span>
      {stat.significant ? <span className="ml-1 text-amber">✱</span> : null}
    </span>
  );
}

function QualityStat({ stat }: { stat?: PairedStat | null }) {
  if (!stat) return <span className="text-edge">—</span>;
  const tone = stat.mean_delta >= 0 ? "text-lime" : "text-crimson";
  return (
    <span className="font-mono text-3xs tabular-nums">
      <span className={tone}>
        {stat.mean_delta >= 0 ? "+" : ""}
        {stat.mean_delta.toFixed(3)}
      </span>
      <span className="text-edge"> ±{stat.stderr.toFixed(3)}</span>
      {stat.significant ? <span className="ml-1 text-amber">✱</span> : null}
    </span>
  );
}

export default function ComparePage() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ExperimentComparison | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .experiments()
      .then((list) => {
        setExperiments(list);
        const fromUrl = new URLSearchParams(window.location.search).get("experiment");
        setSelected(fromUrl ?? list[0]?.experiment ?? null);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const load = useCallback((name: string) => {
    api.experiment(name).then(setComparison).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (selected) load(selected);
  }, [selected, load]);

  return (
    <main className="mx-auto max-w-[90rem] space-y-3 p-3">
      <header className="panel flex flex-wrap items-baseline justify-between gap-2 px-4 py-2">
        <div>
          <h1 className="font-pixel text-xs text-phosphor">IS ORCHESTRATION WORTH IT</h1>
          <p className="mt-1 font-mono text-3xs text-[#8f89c9]">
            Every arm ran the same task. Differences are paired on shared seeds.
          </p>
        </div>
        <nav className="flex items-center gap-4">
          <SiteNav current="/compare" />
        </nav>
      </header>

      {error ? (
        <div className="panel border-crimson px-4 py-3 font-mono text-3xs text-crimson">{error}</div>
      ) : null}

      {experiments.length === 0 ? (
        <div className="panel px-4 py-8 text-center">
          <p className="font-mono text-xs text-edge">
            No experiments yet. Create runs with an <code>experiment</code> and an{" "}
            <code>arm</code>, including one arm named <code>control</code>.
          </p>
        </div>
      ) : null}

      {experiments.length > 0 ? (
        <div className="panel flex flex-wrap items-center gap-2 px-3 py-2">
          <span className="stat-label">Experiment</span>
          {experiments.map((entry) => (
            <button
              key={entry.experiment}
              type="button"
              onClick={() => setSelected(entry.experiment)}
              className={`pixel-btn ${selected === entry.experiment ? "pixel-btn-primary" : ""}`}
            >
              {entry.experiment}
              {!entry.has_control ? <span className="ml-1 text-crimson">!</span> : null}
            </button>
          ))}
        </div>
      ) : null}

      {comparison ? (
        <>
          <section className="panel px-4 py-3">
            <div className="flex items-baseline justify-between gap-3">
              <p className="stat-label">Verdict</p>
              <span
                className={`border-2 px-2 py-0.5 font-pixel text-3xs ${
                  comparison.mode === "live"
                    ? "border-lime text-lime"
                    : comparison.mode === "mixed"
                      ? "border-crimson text-crimson"
                      : "border-amber text-amber"
                }`}
              >
                {comparison.mode === "live"
                  ? "LIVE AGENTS"
                  : comparison.mode === "mixed"
                    ? "MIXED MODES"
                    : "SIMULATED"}
              </span>
            </div>
            <p className="mt-1 font-mono text-xs leading-relaxed text-phosphor">
              {comparison.verdict}
            </p>
            <p className="mt-2 font-mono text-3xs text-edge">
              Task: {comparison.tasks.join(" · ")}
            </p>
          </section>

          <section className="panel overflow-x-auto px-4 py-3">
            <table className="w-full border-collapse font-mono text-3xs">
              <thead>
                <tr className="text-left">
                  {[
                    "arm",
                    "seeds",
                    "quality",
                    "cost",
                    "tokens",
                    "steps",
                    "esc",
                    "Δcost",
                    "Δtokens",
                    "Δquality",
                  ].map((h) => (
                    <th key={h} className="border-y-2 border-edge px-2 py-1 stat-label">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {comparison.arms.map((arm) => {
                  const isControl = arm.arm === comparison.control_arm;
                  return (
                    <tr
                      key={arm.arm}
                      className={`border-b border-edge/40 ${isControl ? "bg-slabLight" : ""}`}
                    >
                      <td className="px-2 py-1">
                        <span className={isControl ? "text-amber" : "text-phosphor"}>
                          {arm.arm}
                        </span>
                        <span className="ml-2 text-edge">{arm.policy}</span>
                      </td>
                      <td className="px-2 py-1 text-edge">{arm.seeds.length}</td>
                      <td className="px-2 py-1 text-cyan">
                        {arm.mean_quality === null ? (
                          <span className="text-crimson">unjudged</span>
                        ) : (
                          arm.mean_quality.toFixed(2)
                        )}
                      </td>
                      <td className="px-2 py-1 text-violet">${arm.mean_cost_usd.toFixed(4)}</td>
                      <td className="px-2 py-1 text-edge">
                        {Math.round(arm.mean_tokens).toLocaleString()}
                      </td>
                      <td className="px-2 py-1 text-edge">{arm.mean_steps.toFixed(1)}</td>
                      <td className="px-2 py-1 text-edge">
                        {arm.escalated}/{arm.runs}
                      </td>
                      <td className="px-2 py-1">
                        <Stat stat={arm.vs_control?.cost_usd} precision={4} />
                      </td>
                      <td className="px-2 py-1">
                        <Stat stat={arm.vs_control?.tokens} precision={0} />
                      </td>
                      <td className="px-2 py-1">
                        <QualityStat stat={arm.vs_control?.quality} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="mt-2 font-mono text-3xs text-edge">
              Δ is this arm minus control on shared seeds. ✱ exceeds two standard errors. Green is
              better: cheaper for cost and tokens, higher for quality.
            </p>
          </section>

          <section className="panel px-4 py-3">
            <p className="stat-label">Read this before believing the table</p>
            <ul className="mt-2 space-y-1">
              {comparison.caveats.map((caveat) => (
                <li key={caveat} className="font-mono text-3xs leading-relaxed text-[#8f89c9]">
                  · {caveat}
                </li>
              ))}
            </ul>
          </section>

          <section className="panel px-4 py-3">
            <p className="stat-label">Runs</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {comparison.arms.flatMap((arm) =>
                arm.run_ids.map((id) => (
                  <Link
                    key={id}
                    href={`/?run=${id}&tab=traces`}
                    className="pixel-btn text-3xs"
                    title={`${arm.arm} · ${id}`}
                  >
                    {arm.arm}/{id.slice(0, 6)}
                  </Link>
                )),
              )}
            </div>
          </section>
        </>
      ) : null}
    </main>
  );
}
