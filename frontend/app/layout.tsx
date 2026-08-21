import type { Metadata, Viewport } from "next";

import "./globals.css";
import { SpriteManifestProvider } from "@/components/pixel/PixelSprite";

export const metadata: Metadata = {
  title: "MARKOV // AGENT ORCHESTRATOR",
  description:
    "A stochastic multi-agent orchestration arena: contextual bandits, MDPs, cooperative Markov games and multi-agent RL, rendered as a pixel-art game.",
};

export const viewport: Viewport = {
  themeColor: "#07060f",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        {/* Loaded via link rather than next/font so an offline build still succeeds. */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Press+Start+2P&family=IBM+Plex+Mono:wght@400;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="crt grid-bg min-h-screen bg-void text-[#d8d4ff] antialiased">
        <SpriteManifestProvider>{children}</SpriteManifestProvider>
      </body>
    </html>
  );
}
