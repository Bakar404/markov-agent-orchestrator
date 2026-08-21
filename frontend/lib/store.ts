"use client";

import { create } from "zustand";

import { ApiError, api, runSocketUrl } from "./api";
import type {
  CreateRunPayload,
  Meta,
  RunDetail,
  RunMetrics,
  SocketEvent,
  StepResult,
  Trace,
} from "./types";

export type Phase = "title" | "booting" | "arena";

/** Kept outside the store: a live socket is not serializable state. */
let socket: WebSocket | null = null;

const MAX_FEED = 60;

interface GameState {
  phase: Phase;
  meta: Meta | null;
  run: RunDetail | null;
  lastStep: StepResult | null;
  feed: StepResult[];
  metrics: RunMetrics | null;
  traces: Trace[];
  playing: boolean;
  intervalMs: number;
  connected: boolean;
  error: string | null;
  booting: boolean;
  /** Agents rendered mid-action this tick, used to drive sprite animation. */
  activeAgents: string[];
  combo: number;

  loadMeta: () => Promise<void>;
  startGame: (payload: CreateRunPayload) => Promise<void>;
  connect: (runId: string) => void;
  play: () => void;
  pause: () => void;
  stepOnce: () => void;
  reset: (seed?: number, keepPolicyLearning?: boolean) => void;
  setSpeed: (intervalMs: number) => void;
  refreshAnalytics: () => Promise<void>;
  quitToTitle: () => void;
  clearError: () => void;
}

export const useGame = create<GameState>((set, get) => ({
  phase: "title",
  meta: null,
  run: null,
  lastStep: null,
  feed: [],
  metrics: null,
  traces: [],
  playing: false,
  intervalMs: 750,
  connected: false,
  error: null,
  booting: false,
  activeAgents: [],
  combo: 0,

  clearError: () => set({ error: null }),

  loadMeta: async () => {
    try {
      set({ meta: await api.meta(), error: null });
    } catch (error) {
      set({ error: error instanceof ApiError ? error.message : String(error) });
    }
  },

  startGame: async (payload) => {
    set({ booting: true, error: null, phase: "booting" });
    try {
      const run = await api.createRun(payload);
      set({
        run,
        phase: "arena",
        booting: false,
        feed: [],
        lastStep: null,
        metrics: null,
        traces: [],
        combo: 0,
        activeAgents: [],
      });
      get().connect(run.id);
      void get().refreshAnalytics();
    } catch (error) {
      set({
        booting: false,
        phase: "title",
        error: error instanceof ApiError ? error.message : String(error),
      });
    }
  },

  connect: (runId) => {
    socket?.close();
    const next = new WebSocket(runSocketUrl(runId));
    socket = next;

    next.onopen = () => set({ connected: true, error: null });
    next.onclose = () => set({ connected: false, playing: false });
    next.onerror = () =>
      set({ error: "Lost the connection to the orchestration socket.", connected: false });

    next.onmessage = (event) => {
      const payload = JSON.parse(event.data as string) as SocketEvent;

      if (payload.type === "snapshot") {
        set({ run: payload.run, feed: [], lastStep: null, combo: 0, activeAgents: [] });
        void get().refreshAnalytics();
        return;
      }

      if (payload.type === "step") {
        const step = payload.step;
        const previous = get().combo;
        set((state) => ({
          run: payload.run,
          lastStep: step,
          feed: [step, ...state.feed].slice(0, MAX_FEED),
          activeAgents: step.agents,
          combo: step.reward > 0 ? previous + 1 : 0,
        }));
        return;
      }

      if (payload.type === "status") {
        set({ playing: payload.playing, intervalMs: payload.interval_ms });
        return;
      }

      if (payload.type === "terminated") {
        set({ run: payload.run, playing: false, activeAgents: [] });
        void get().refreshAnalytics();
        return;
      }

      if (payload.type === "error") {
        set({ error: payload.detail });
      }
    };
  },

  play: () => {
    if (socket?.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({ type: "start", interval_ms: get().intervalMs }));
    set({ playing: true });
  },

  pause: () => {
    if (socket?.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({ type: "pause" }));
    set({ playing: false });
    void get().refreshAnalytics();
  },

  stepOnce: () => {
    if (socket?.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({ type: "step" }));
  },

  reset: (seed, keepPolicyLearning = false) => {
    if (socket?.readyState !== WebSocket.OPEN) return;
    socket.send(
      JSON.stringify({
        type: "reset",
        seed: seed ?? null,
        keep_policy_learning: keepPolicyLearning,
      }),
    );
    set({ playing: false, feed: [], lastStep: null, combo: 0, activeAgents: [] });
  },

  setSpeed: (intervalMs) => {
    set({ intervalMs });
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "speed", interval_ms: intervalMs }));
    }
  },

  refreshAnalytics: async () => {
    const run = get().run;
    if (!run) return;
    try {
      const [metrics, traces] = await Promise.all([api.metrics(run.id), api.traces(run.id)]);
      set({ metrics, traces });
    } catch {
      // Analytics are supplementary; a failure here must not interrupt playback.
    }
  },

  quitToTitle: () => {
    socket?.close();
    socket = null;
    set({
      phase: "title",
      run: null,
      lastStep: null,
      feed: [],
      metrics: null,
      traces: [],
      playing: false,
      connected: false,
      activeAgents: [],
      combo: 0,
    });
  },
}));
