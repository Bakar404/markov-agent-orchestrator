"use client";

import Link from "next/link";

const LINKS = [
  { href: "/", label: "ARENA" },
  { href: "/compare", label: "COMPARE" },
] as const;

/** One nav for the two questions a run answers. `/research` still resolves if typed. */
export function SiteNav({ current }: { current?: string }) {
  return (
    <nav className="flex items-center gap-1">
      {LINKS.map((link) => {
        const active = link.href === current;
        return (
          <Link
            key={link.href}
            href={link.href}
            className={`border-2 px-3 py-1 font-pixel text-3xs ${
              active
                ? "border-phosphor bg-phosphor text-void"
                : "border-edge bg-slab text-edge hover:border-phosphor hover:text-phosphor"
            }`}
          >
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}
