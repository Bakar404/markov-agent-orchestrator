import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Arcade CRT palette
        void: "#07060f",
        panel: "#12102080",
        ink: "#0b0a14",
        slab: "#171531",
        slabLight: "#221f42",
        edge: "#3b356b",
        phosphor: "#7bf7c4",
        magenta: "#ff5fd2",
        amber: "#ffc857",
        cyan: "#5fe3ff",
        violet: "#a78bfa",
        crimson: "#ff5f6d",
        lime: "#b8ff5f",
        agent: {
          planner: "#38bdf8",
          researcher: "#a78bfa",
          critic: "#fb7185",
          verifier: "#34d399",
          memory: "#fbbf24",
          executor: "#f472b6",
          orchestrator: "#7bf7c4",
        },
      },
      fontFamily: {
        pixel: ["'Press Start 2P'", "'Courier New'", "monospace"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "Consolas", "monospace"],
      },
      fontSize: {
        "2xs": ["0.5rem", { lineHeight: "1.1rem" }],
        "3xs": ["0.4rem", { lineHeight: "0.9rem" }],
      },
      boxShadow: {
        pixel: "4px 4px 0 0 #07060f",
        "pixel-sm": "2px 2px 0 0 #07060f",
        "pixel-lg": "6px 6px 0 0 #07060f",
        glow: "0 0 12px 0 rgba(123,247,196,0.55)",
        "glow-magenta": "0 0 12px 0 rgba(255,95,210,0.55)",
      },
      animation: {
        blink: "blink 1.05s steps(2, start) infinite",
        bob: "bob 1.4s steps(4, end) infinite",
        "scan-roll": "scan-roll 8s linear infinite",
        "pulse-ring": "pulse-ring 1.2s ease-out infinite",
        marquee: "marquee 22s linear infinite",
        shake: "shake 0.32s steps(3, end) 1",
        "pop-in": "pop-in 0.22s steps(3, end) 1",
      },
      keyframes: {
        blink: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
        bob: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-4px)" },
        },
        "scan-roll": {
          "0%": { backgroundPosition: "0 0" },
          "100%": { backgroundPosition: "0 -100vh" },
        },
        "pulse-ring": {
          "0%": { transform: "scale(0.85)", opacity: "0.9" },
          "100%": { transform: "scale(1.5)", opacity: "0" },
        },
        marquee: {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        shake: {
          "0%, 100%": { transform: "translateX(0)" },
          "33%": { transform: "translateX(-3px)" },
          "66%": { transform: "translateX(3px)" },
        },
        "pop-in": {
          "0%": { transform: "scale(0.6)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
