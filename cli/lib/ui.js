/**
 * The arena's palette, rendered as ANSI. Colours match tailwind.config.ts so the terminal and
 * the browser are recognisably the same product.
 */

const truecolor = (hex) => {
  const n = parseInt(hex.slice(1), 16);
  return `\x1b[38;2;${(n >> 16) & 255};${(n >> 8) & 255};${n & 255}m`;
};

const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";

// Straight from tailwind.config.ts.
const HEX = {
  phosphor: "#7bf7c4",
  magenta: "#ff5fd2",
  amber: "#ffc857",
  cyan: "#5fe3ff",
  violet: "#a78bfa",
  crimson: "#ff5f6d",
  lime: "#b8ff5f",
  mist: "#e6e3ff",
  edge: "#3b356b",
  planner: "#38bdf8",
  researcher: "#a78bfa",
  critic: "#fb7185",
  verifier: "#34d399",
  memory: "#fbbf24",
  executor: "#f472b6",
  generalist: "#7bf7c4",
};

const enabled = process.env.NO_COLOR === undefined && process.stdout.isTTY !== false;
const wrap = (hex) => (text) => (enabled ? `${truecolor(hex)}${text}${RESET}` : String(text));

export const c = Object.fromEntries(Object.entries(HEX).map(([k, v]) => [k, wrap(v)]));
c.bold = (t) => (enabled ? `${BOLD}${t}${RESET}` : String(t));
c.dim = (t) => (enabled ? `${DIM}${t}${RESET}` : String(t));

export const AGENT_COLOR = {
  planner: c.planner,
  researcher: c.researcher,
  critic: c.critic,
  verifier: c.verifier,
  memory: c.memory,
  executor: c.executor,
  generalist: c.generalist,
};

/** Visible width, ignoring the escape sequences. */
export const width = (s) => s.replace(/\x1b\[[0-9;]*m/g, "").length;

const pad = (s, n) => s + " ".repeat(Math.max(0, n - width(s)));

export function panel(title, lines, { color = c.phosphor, inner } = {}) {
  // Auto-size to the widest line so long help text cannot break the right border.
  const span = inner ?? Math.max(width(title) + 4, ...lines.map(width)) + 1;
  const top = color(`┌─ ${title} ` + "─".repeat(Math.max(0, span - width(title) - 3)) + "┐");
  const bottom = color("└" + "─".repeat(span + 1) + "┘");
  const body = lines.map((l) => `${color("│")} ${pad(l, span - 1)}${color("│")}`);
  return [top, ...body, bottom].join("\n");
}

export function rule(label = "", color = c.edge) {
  const line = "─".repeat(Math.max(0, 68 - width(label) - (label ? 2 : 0)));
  return color(label ? `── ${label} ${line}` : `──${line}──`);
}

/** The attract screen, in the same shape the browser shows. */
export function banner() {
  const art = [
    "  ▄▄▄   ▄▄▄▄  ▄▄▄▄▄ ▄▄   ▄  ▄▄▄ ",
    " ▐█ ██▌ ▐█ ▐█ ▐█    ▐██▄ █ ▐█ ▐█",
    " ▐████▌ ▐███▌ ▐███  ▐█ ██▌ ▐████",
    " ▐█  █▌ ▐█ █▌ ▐█    ▐█  █▌ ▐█ ▐█",
    " ▐█  █▌ ▐█ ▐█ ▐████ ▐█  █▌ ▐█ ▐█",
  ];
  return [
    "",
    ...art.map((l) => c.phosphor(l)),
    c.magenta("  IS ORCHESTRATION WORTH IT"),
    "",
  ].join("\n");
}

/** A sprite row: one glyph per agent, dimmed until it has acted. */
export function roster(active = [], unlocked = false) {
  const all = ["generalist", "planner", "researcher", "critic", "verifier", "memory", "executor"];
  return all
    .map((id) => {
      const glyph = id === "generalist" ? "◈" : "◆";
      if (active.includes(id)) return AGENT_COLOR[id](`${glyph}`);
      if (!unlocked && id !== "generalist") return c.dim("·");
      return c.edge(glyph);
    })
    .join(" ");
}

export const ok = (t) => c.phosphor(`✓ ${t}`);
export const warn = (t) => c.amber(`! ${t}`);
export const bad = (t) => c.crimson(`✗ ${t}`);

/** Hard-wrap to a column so a long verdict cannot produce a 190-wide panel. */
export function wrapText(text, span = 66) {
  const lines = [];
  for (const paragraph of String(text).split("\n")) {
    let line = "";
    for (const word of paragraph.split(/\s+/)) {
      if (line && `${line} ${word}`.length > span) {
        lines.push(line);
        line = word;
      } else {
        line = line ? `${line} ${word}` : word;
      }
    }
    lines.push(line);
  }
  return lines;
}
