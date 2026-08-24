"use client";

import { useEffect } from "react";

import { isLive, useGame } from "@/lib/store";

const SPEEDS = [
  { label: "0.5x", ms: 1400 },
  { label: "1x", ms: 750 },
  { label: "2x", ms: 350 },
  { label: "4x", ms: 140 },
];

export function Controls() {
  const {
    run,
    playing,
    intervalMs,
    connected,
    play,
    pause,
    stepOnce,
    reset,
    setSpeed,
    quitToTitle,
    refreshAnalytics,
  } = useGame();

  const done = Boolean(run?.terminated);
  const live = isLive({ run });

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA"].includes(target.tagName)) return;

      if (event.key.toLowerCase() === "r") reset();
      if (live) return;

      if (event.code === "Space") {
        event.preventDefault();
        if (done) return;
        playing ? pause() : play();
      }
      if (event.code === "ArrowRight") {
        event.preventDefault();
        if (!done && !playing) stepOnce();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [playing, done, live, play, pause, stepOnce, reset]);

  return (
    <div className="panel flex flex-wrap items-center gap-2 px-3 py-2">
      {live ? (
        <div className="flex items-center gap-2 border-2 border-violet px-2 py-1">
          <span className="font-pixel text-3xs text-violet">◆ LIVE</span>
          <span className="font-mono text-3xs text-edge">
            driven by agent chat · steps arrive as they are reported
          </span>
        </div>
      ) : (
        <>
          <button
            type="button"
            className={`pixel-btn ${playing ? "" : "pixel-btn-primary"}`}
            disabled={!connected || done}
            onClick={() => (playing ? pause() : play())}
          >
            {playing ? "❚❚ Pause" : "▶ Start"}
          </button>

          <button
            type="button"
            className="pixel-btn"
            disabled={!connected || done || playing}
            onClick={stepOnce}
          >
            ▶❙ Step
          </button>
        </>
      )}

      <button type="button" className="pixel-btn" disabled={!connected} onClick={() => reset()}>
        ↺ Reset
      </button>

      {!live && (
        <div className="flex items-center gap-1 border-2 border-edge px-2 py-1">
          <span className="stat-label">Speed</span>
          {SPEEDS.map((speed) => (
            <button
              key={speed.ms}
              type="button"
              onClick={() => setSpeed(speed.ms)}
              className={`px-1 font-pixel text-3xs ${
                intervalMs === speed.ms ? "text-phosphor" : "text-edge hover:text-violet"
              }`}
            >
              {speed.label}
            </button>
          ))}
        </div>
      )}

      <button type="button" className="pixel-btn" onClick={() => void refreshAnalytics()}>
        ⟳ Stats
      </button>

      <div className="ml-auto flex items-center gap-3">
        <span className="font-pixel text-3xs" style={{ color: connected ? "#b8ff5f" : "#ff5f6d" }}>
          {connected ? "● LINK OK" : "● NO LINK"}
        </span>
        <button type="button" className="pixel-btn pixel-btn-danger" onClick={quitToTitle}>
          Quit
        </button>
      </div>

      <p className="w-full font-mono text-3xs text-edge">
        {live ? "R reset · drive the run from your agent chat" : "SPACE start/pause · → step · R reset"}
      </p>
    </div>
  );
}
