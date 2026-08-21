"use client";

import { useEffect, useMemo, useState } from "react";

import { PixelSprite } from "@/components/pixel/PixelSprite";
import { useGame } from "@/lib/store";
import type { AgentSpec } from "@/lib/types";

const RING_ORDER = ["planner", "researcher", "critic", "verifier", "memory", "executor"];

interface FloatingNumber {
  key: string;
  agentId: string;
  value: number;
}

/** Positions the six agents evenly on a ring around the core. */
function ringPosition(index: number, total: number) {
  const angle = (index / total) * Math.PI * 2 - Math.PI / 2;
  return {
    left: `${50 + Math.cos(angle) * 34}%`,
    top: `${50 + Math.sin(angle) * 34}%`,
  };
}

export function Arena() {
  const { run, lastStep, activeAgents, combo } = useGame();
  const [floats, setFloats] = useState<FloatingNumber[]>([]);

  const agents = useMemo(() => {
    const byId = new Map((run?.agents ?? []).map((a) => [a.id, a]));
    return RING_ORDER.map((id) => byId.get(id)).filter(Boolean) as AgentSpec[];
  }, [run?.agents]);

  // Spawn a damage-number style readout for each agent that acted this step.
  useEffect(() => {
    if (!lastStep) return;
    const shares = lastStep.reward_breakdown.per_agent ?? {};
    const spawned = lastStep.agents.map((agentId) => ({
      key: `${lastStep.step}-${agentId}`,
      agentId,
      value: shares[agentId] ?? lastStep.reward / Math.max(lastStep.agents.length, 1),
    }));
    setFloats((current) => [...current, ...spawned].slice(-12));
    const timer = setTimeout(() => {
      setFloats((current) => current.filter((f) => !spawned.some((s) => s.key === f.key)));
    }, 1100);
    return () => clearTimeout(timer);
  }, [lastStep]);

  const state = run?.state;
  const fog = state ? state.uncertainty : 1;
  const coreHealth = state ? state.confidence : 0;

  return (
    <div className="panel relative h-full min-h-[26rem] overflow-hidden">
      {/* Entropy fog: the arena literally clears as uncertainty collapses. */}
      <div
        className="pointer-events-none absolute inset-0 z-20 bg-void transition-opacity duration-500"
        style={{ opacity: fog * 0.55 }}
      />
      <div className="dither pointer-events-none absolute inset-0 opacity-40" />

      <div className="absolute left-3 top-3 z-30">
        <p className="stat-label">Entropy fog</p>
        <p className="font-pixel text-2xs text-cyan">{(fog * 100).toFixed(0)}%</p>
      </div>

      {combo > 1 ? (
        <div className="absolute right-3 top-3 z-30 animate-pop-in text-right">
          <p className="stat-label">Combo</p>
          <p className="font-pixel text-sm text-amber">x{combo}</p>
        </div>
      ) : null}

      {/* Core */}
      <div className="absolute left-1/2 top-1/2 z-10 -translate-x-1/2 -translate-y-1/2 text-center">
        <div className="relative">
          {activeAgents.length > 0 ? (
            <span className="absolute inset-0 animate-pulse-ring border-2 border-phosphor" />
          ) : null}
          <PixelSprite id="orchestrator" size={96} animation="idle" />
        </div>
        <p className="mt-1 font-pixel text-3xs text-phosphor">CORE</p>
        <div className="meter mx-auto mt-1 w-24">
          <div
            className="meter-fill text-phosphor"
            style={{ width: `${Math.round(coreHealth * 100)}%` }}
          />
        </div>
      </div>

      {/* Agents on the ring */}
      {agents.map((agent, index) => {
        const active = activeAgents.includes(agent.id);
        const history = state?.agent_history?.[agent.id];
        const float = floats.find((f) => f.agentId === agent.id);
        return (
          <div
            key={agent.id}
            className="absolute z-30 -translate-x-1/2 -translate-y-1/2 text-center"
            style={ringPosition(index, agents.length)}
          >
            {float ? (
              <span
                className={`absolute -top-6 left-1/2 -translate-x-1/2 animate-pop-in font-pixel text-2xs ${
                  float.value >= 0 ? "text-lime" : "text-crimson"
                }`}
              >
                {float.value >= 0 ? "+" : ""}
                {float.value.toFixed(2)}
              </span>
            ) : null}

            <div className={active ? "animate-bob" : ""}>
              <PixelSprite
                id={agent.id}
                size={active ? 72 : 56}
                dim={!active && (history?.invocations ?? 0) === 0}
                title={agent.label}
                animation={active ? "attack" : "idle"}
              />
            </div>

            <p
              className="font-pixel text-3xs"
              style={{ color: active ? agent.color : "#5b5691" }}
            >
              {agent.label.replace(" Agent", "").toUpperCase()}
            </p>

            {history ? (
              <p className="font-mono text-3xs text-edge">
                x{history.invocations} · {(history.success_rate * 100).toFixed(0)}%
              </p>
            ) : null}
          </div>
        );
      })}

      {lastStep ? (
        <div className="absolute bottom-0 left-0 right-0 z-30 border-t-2 border-edge bg-ink/90 px-3 py-2">
          <p className="font-mono text-3xs text-[#8f89c9]">
            <span className="text-amber">STEP {lastStep.step}</span> · {lastStep.action_label} ·{" "}
            <span
              className={
                lastStep.outcome === "success"
                  ? "text-lime"
                  : lastStep.outcome === "failure"
                    ? "text-crimson"
                    : "text-amber"
              }
            >
              {lastStep.outcome.toUpperCase()}
            </span>{" "}
            · p(a)={lastStep.action_probability.toFixed(3)} · ΔH=
            {lastStep.information_gain >= 0 ? "+" : ""}
            {lastStep.information_gain.toFixed(3)} bits
          </p>
        </div>
      ) : null}
    </div>
  );
}
