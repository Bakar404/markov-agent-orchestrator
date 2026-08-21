/**
 * Procedural pixel sprites.
 *
 * These 16x16 grids are the fallback so the game is playable before any art is generated.
 * When PixelLab output is dropped into `public/sprites/` and listed in
 * `public/sprites/manifest.json`, `PixelSprite` swaps to the PNG automatically and these
 * grids stay as the offline fallback.
 *
 * Grid legend:
 *   `.` transparent   `k` outline   `1` highlight   `2` body   `3` shadow
 *   `4` accent        `w` white
 */

export type Palette = Record<string, string>;

export interface SpriteDefinition {
  id: string;
  label: string;
  grid: string[];
  palette: Palette;
}

const OUTLINE = "#05040c";
const WHITE = "#ffffff";

/** Rows 3..15 of the standard agent body. Hats occupy rows 0..2. */
const BODY: string[] = [
  "...k111111k.....",
  "...k1w11w1k.....",
  "...k111111k.....",
  "...k1kkkk1k.....",
  "....kk11kk......",
  "...k222222k.....",
  "..k22222222k....",
  "..k2k2222k2k....",
  "..k2k2222k2k....",
  "...k222222k.....",
  "....k33k33k.....",
  "....k3k.k3k.....",
  "....kkk.kkk.....",
];

const HATS: Record<string, string[]> = {
  // Hard hat with a wide brim
  planner: ["................", "....k4444k......", "..kk444444kk...."],
  // Tall pointed wizard hat
  researcher: ["......k4k.......", ".....k444k......", "...k44444444k..."],
  // Horned visor helm
  critic: ["...k........k...", "...k4......4k...", "...k44444444k..."],
  // Crested helm
  verifier: ["................", ".....k444k......", "...k44444444k..."],
  // Antenna with an orb
  memory: ["......k4k.......", ".......k........", "....kkkkkk......"],
  // Welding mask
  executor: ["................", "...kk4444kk.....", "...k444444k....."],
};

const PALETTES: Record<string, Palette> = {
  planner: { k: OUTLINE, w: WHITE, "1": "#7dd3fc", "2": "#38bdf8", "3": "#0369a1", "4": "#fbbf24" },
  researcher: { k: OUTLINE, w: WHITE, "1": "#c4b5fd", "2": "#a78bfa", "3": "#5b21b6", "4": "#312e81" },
  critic: { k: OUTLINE, w: WHITE, "1": "#fda4af", "2": "#fb7185", "3": "#9f1239", "4": "#f43f5e" },
  verifier: { k: OUTLINE, w: WHITE, "1": "#6ee7b7", "2": "#34d399", "3": "#047857", "4": "#a7f3d0" },
  memory: { k: OUTLINE, w: WHITE, "1": "#fde68a", "2": "#fbbf24", "3": "#b45309", "4": "#fef3c7" },
  executor: { k: OUTLINE, w: WHITE, "1": "#f9a8d4", "2": "#f472b6", "3": "#9d174d", "4": "#64748b" },
  orchestrator: { k: OUTLINE, w: WHITE, "1": "#7bf7c4", "2": "#34d399", "3": "#047857", "4": "#0f766e" },
};

const LABELS: Record<string, string> = {
  planner: "Planner",
  researcher: "Research",
  critic: "Critic",
  verifier: "Verifier",
  memory: "Memory",
  executor: "Executor",
  orchestrator: "Core",
};

/** The orchestrator is a floating crystal core rather than a humanoid. */
const ORCHESTRATOR_GRID: string[] = [
  "................",
  "................",
  ".......kk.......",
  "......k44k......",
  ".....k4114k.....",
  "....k411114k....",
  "...k41111114k...",
  "..k4111ww1114k..",
  "..k4111ww1114k..",
  "...k41111114k...",
  "....k411114k....",
  ".....k4114k.....",
  "......k44k......",
  ".......kk.......",
  "................",
  "................",
];

function buildAgent(id: string): SpriteDefinition {
  return {
    id,
    label: LABELS[id] ?? id,
    grid: [...HATS[id], ...BODY],
    palette: PALETTES[id],
  };
}

export const SPRITES: Record<string, SpriteDefinition> = {
  planner: buildAgent("planner"),
  researcher: buildAgent("researcher"),
  critic: buildAgent("critic"),
  verifier: buildAgent("verifier"),
  memory: buildAgent("memory"),
  executor: buildAgent("executor"),
  orchestrator: {
    id: "orchestrator",
    label: LABELS.orchestrator,
    grid: ORCHESTRATOR_GRID,
    palette: PALETTES.orchestrator,
  },
};

export const SPRITE_SIZE = 16;

export function getSprite(id: string): SpriteDefinition {
  return SPRITES[id] ?? SPRITES.orchestrator;
}

/**
 * Contents of `public/sprites/manifest.json`. Generated art is opt-in per sprite id, so a
 * partially generated set degrades to procedural art for the ids that are still missing.
 */
export interface SpriteAnimation {
  /** Explicit frame URLs in play order. No filename convention to guess at. */
  frames: string[];
  fps: number;
  /** Play once and fall back to idle, rather than looping. */
  once?: boolean;
}

export interface SpriteEntry {
  src: string;
  size?: number;
  description?: string;
  animations?: Record<string, SpriteAnimation>;
}

export interface SpriteManifest {
  generator?: string;
  generatedAt?: string;
  note?: string;
  sprites: Record<string, SpriteEntry>;
}

export const EMPTY_MANIFEST: SpriteManifest = { sprites: {} };
