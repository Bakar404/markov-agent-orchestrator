"use client";

import { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";

import {
  EMPTY_MANIFEST,
  SPRITE_SIZE,
  type SpriteManifest,
  getSprite,
} from "@/lib/sprites";

const ManifestContext = createContext<SpriteManifest>(EMPTY_MANIFEST);

export function SpriteManifestProvider({ children }: { children: React.ReactNode }) {
  const [manifest, setManifest] = useState<SpriteManifest>(EMPTY_MANIFEST);

  useEffect(() => {
    let cancelled = false;
    fetch("/sprites/manifest.json", { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : EMPTY_MANIFEST))
      .then((data: SpriteManifest) => {
        if (!cancelled && data && typeof data === "object") {
          setManifest({ ...EMPTY_MANIFEST, ...data, sprites: data.sprites ?? {} });
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  return <ManifestContext.Provider value={manifest}>{children}</ManifestContext.Provider>;
}

export function useSpriteManifest(): SpriteManifest {
  return useContext(ManifestContext);
}

export type SpriteAnimationName = "idle" | "attack";

interface PixelSpriteProps {
  id: string;
  /** Rendered edge length in CSS pixels. */
  size?: number;
  className?: string;
  title?: string;
  /** Dim the sprite when the agent is idle. */
  dim?: boolean;
  /** Directional variant, when generated art provides one. */
  facing?: "south" | "east" | "north" | "west";
  /** Animation to play. "attack" runs once then falls back to "idle". */
  animation?: SpriteAnimationName;
}

export function PixelSprite({
  id,
  size = 64,
  className = "",
  title,
  dim = false,
  facing = "south",
  animation,
}: PixelSpriteProps) {
  const manifest = useSpriteManifest();
  const definition = getSprite(id);
  const generated = manifest.sprites?.[id];

  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState<SpriteAnimationName | undefined>(animation);
  const previous = useRef<SpriteAnimationName | undefined>(animation);

  // Restart playback whenever the requested animation changes, so a repeated attack replays.
  useEffect(() => {
    if (animation !== previous.current) {
      previous.current = animation;
      setPlaying(animation);
      setFrame(0);
    }
  }, [animation]);

  const clip = playing ? generated?.animations?.[playing] : undefined;

  useEffect(() => {
    if (!clip || clip.frames.length < 2) return;
    const interval = setInterval(
      () => {
        setFrame((current) => {
          const next = current + 1;
          if (next < clip.frames.length) return next;
          if (clip.once) {
            // One-shot clips hand control back to the looping idle.
            setPlaying("idle");
            return 0;
          }
          return 0;
        });
      },
      1000 / Math.max(clip.fps, 1),
    );
    return () => clearInterval(interval);
  }, [clip]);

  // Procedural grids are 16x16 and snap to a whole multiple to stay crisp. Generated art has
  // its own native size and is left on the requested box, with pixelated rendering doing the
  // nearest-neighbour work.
  const scale = Math.max(1, Math.round(size / SPRITE_SIZE));
  const edge = generated ? size : scale * SPRITE_SIZE;

  const rects = useMemo(() => {
    if (generated) return [];
    const output: { x: number; y: number; fill: string }[] = [];
    definition.grid.forEach((row, y) => {
      for (let x = 0; x < row.length; x += 1) {
        const key = row[x];
        if (key === ".") continue;
        const fill = definition.palette[key];
        if (fill) output.push({ x, y, fill });
      }
    });
    return output;
  }, [definition, generated]);

  const style = { width: edge, height: edge, opacity: dim ? 0.45 : 1 };

  if (generated && clip && clip.frames.length > 0) {
    // Every frame stays mounted so the browser keeps them decoded; only one is visible.
    // Swapping a single src instead flashes white on each frame change.
    return (
      <span
        style={{ ...style, position: "relative", display: "inline-block" }}
        className={className}
      >
        {clip.frames.map((src, index) => (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={src}
            src={src}
            alt={index === 0 ? (title ?? definition.label) : ""}
            width={edge}
            height={edge}
            draggable={false}
            style={{
              position: "absolute",
              inset: 0,
              visibility: index === frame ? "visible" : "hidden",
            }}
          />
        ))}
      </span>
    );
  }

  if (generated) {
    const src =
      facing === "south" ? generated.src : generated.src.replace(/\.png$/, `-${facing}.png`);
    return (
      // Generated frames are plain rasters; next/image would resample them.
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={src}
        alt={title ?? definition.label}
        width={edge}
        height={edge}
        style={style}
        className={className}
        draggable={false}
      />
    );
  }

  return (
    <svg
      viewBox={`0 0 ${SPRITE_SIZE} ${SPRITE_SIZE}`}
      style={style}
      className={className}
      shapeRendering="crispEdges"
      role="img"
      aria-label={title ?? definition.label}
    >
      {rects.map((rect) => (
        <rect
          key={`${rect.x}-${rect.y}`}
          x={rect.x}
          y={rect.y}
          width={1}
          height={1}
          fill={rect.fill}
        />
      ))}
    </svg>
  );
}
