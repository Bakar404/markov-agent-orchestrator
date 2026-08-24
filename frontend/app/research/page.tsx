"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { api } from "@/lib/api";
import { SiteNav } from "@/components/game/SiteNav";
import type {
  Paper,
  PaperDetail,
  ProviderHealth,
  ResearchStats,
  SearchResponse,
} from "@/lib/types";

function RelevanceBar({ value }: { value: number }) {
  return (
    <div className="meter h-2 w-16">
      <div className="meter-fill text-phosphor" style={{ width: `${value * 100}%` }} />
    </div>
  );
}

function PaperRow({
  paper,
  onSelect,
  selected,
}: {
  paper: Paper;
  onSelect: (id: string) => void;
  selected: boolean;
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(paper.id)}
      className={`w-full border-2 px-3 py-2 text-left ${
        selected ? "border-phosphor bg-phosphor/10" : "border-edge bg-ink hover:border-violet"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <p className="font-mono text-xs leading-snug text-[#d8d4ff]">{paper.title}</p>
        <RelevanceBar value={paper.relevance} />
      </div>
      <p className="mt-1 font-mono text-3xs text-edge">
        {paper.year ?? "n.d."} · {paper.venue || paper.source} ·{" "}
        {paper.authors.slice(0, 3).join(", ")}
        {paper.authors.length > 3 ? " et al." : ""} · {paper.citation_count} citations
      </p>
      <div className="mt-1 flex flex-wrap gap-1">
        {paper.tags.map((tag) => (
          <span key={tag} className="border border-edge px-1 font-mono text-3xs text-violet">
            {tag}
          </span>
        ))}
      </div>
    </button>
  );
}

export default function ResearchPage() {
  const [stats, setStats] = useState<ResearchStats | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [detail, setDetail] = useState<PaperDetail | null>(null);
  const [providers, setProviders] = useState<ProviderHealth[]>([]);
  const [tag, setTag] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadPapers = useCallback(async (activeTag: string | null) => {
    try {
      setPapers(await api.papers({ tag: activeTag ?? undefined, limit: 120 }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    api.researchStats().then(setStats).catch((e) => setError(String(e)));
    api.providers().then(setProviders).catch(() => undefined);
    void loadPapers(null);
  }, [loadPapers]);

  const runSearch = async () => {
    if (query.trim().length < 2) return;
    setSearching(true);
    setError(null);
    try {
      const result = await api.search(query.trim());
      setSearchResult(result);
      setPapers(result.papers);
      setStats(await api.researchStats());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSearching(false);
    }
  };

  const select = async (id: string) => {
    try {
      setDetail(await api.paper(id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <main className="mx-auto max-w-[110rem] space-y-3 p-3">
      <header className="panel flex flex-wrap items-baseline justify-between gap-3 px-4 py-2">
        <div>
          <h1 className="font-pixel text-xs text-phosphor">RESEARCH LIBRARY</h1>
          <p className="mt-1 font-mono text-3xs text-[#8f89c9]">
            {stats
              ? `${stats.papers} papers · ${stats.citations} citation edges · ${stats.tags.length} tags`
              : "Loading…"}
          </p>
        </div>
        <div className="flex items-center gap-4">
          <SiteNav current="/research" />
        </div>
      </header>

      {error ? (
        <div className="panel border-crimson px-4 py-2">
          <p className="font-mono text-xs text-[#ffb3b8]">{error}</p>
        </div>
      ) : null}

      <section className="panel px-4 py-3">
        <div className="flex flex-wrap gap-2">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && void runSearch()}
            placeholder="Search arXiv, Semantic Scholar and the HITS MCP provider…"
            className="min-w-[18rem] flex-1 border-2 border-edge bg-ink px-3 py-2 font-mono text-sm
                       text-phosphor outline-none focus:border-phosphor"
          />
          <button
            type="button"
            className="pixel-btn pixel-btn-primary"
            disabled={searching}
            onClick={() => void runSearch()}
          >
            {searching ? "SEARCHING…" : "SEARCH"}
          </button>
          <button
            type="button"
            className="pixel-btn"
            onClick={() => {
              setSearchResult(null);
              setTag(null);
              void loadPapers(null);
            }}
          >
            CLEAR
          </button>
        </div>

        {searchResult ? (
          <div className="mt-2 flex flex-wrap gap-2">
            {searchResult.providers.map((provider) => (
              <span
                key={provider.provider}
                className="border border-edge px-2 py-1 font-mono text-3xs"
                style={{ color: provider.degraded ? "#ffc857" : "#b8ff5f" }}
                title={provider.error || undefined}
              >
                {provider.provider}: {provider.count}
                {provider.degraded ? " (offline fallback)" : ""} ·{" "}
                {provider.latency_ms.toFixed(0)}ms
              </span>
            ))}
          </div>
        ) : null}
      </section>

      <div className="grid gap-3 lg:grid-cols-[16rem_1fr_22rem]">
        <aside className="space-y-3">
          <section className="panel px-3 py-3">
            <h2 className="font-pixel text-2xs text-phosphor">TAXONOMY</h2>
            <div className="mt-2 space-y-1">
              <button
                type="button"
                onClick={() => {
                  setTag(null);
                  void loadPapers(null);
                }}
                className={`w-full px-2 py-1 text-left font-mono text-3xs ${
                  tag === null ? "bg-phosphor text-void" : "text-[#8f89c9] hover:text-phosphor"
                }`}
              >
                All papers
              </button>
              {(stats?.tags ?? []).map((entry) => (
                <button
                  key={entry.tag}
                  type="button"
                  onClick={() => {
                    setTag(entry.tag);
                    void loadPapers(entry.tag);
                  }}
                  className={`flex w-full items-center justify-between px-2 py-1 text-left font-mono text-3xs ${
                    tag === entry.tag
                      ? "bg-phosphor text-void"
                      : "text-[#8f89c9] hover:text-phosphor"
                  }`}
                >
                  <span>{entry.tag}</span>
                  <span className="text-edge">{entry.count}</span>
                </button>
              ))}
            </div>
          </section>

          <section className="panel px-3 py-3">
            <h2 className="font-pixel text-2xs text-phosphor">PROVIDERS</h2>
            <div className="mt-2 space-y-2">
              {providers.map((provider) => (
                <div key={provider.id}>
                  <p className="font-mono text-3xs text-[#c9c4ff]">
                    <span
                      style={{
                        color: provider.available === false ? "#ff5f6d" : "#b8ff5f",
                      }}
                    >
                      ●
                    </span>{" "}
                    {provider.label}
                  </p>
                  <p className="font-mono text-3xs text-edge">
                    {provider.kind}
                    {provider.mode ? ` · ${provider.mode}` : ""}
                    {provider.records ? ` · ${provider.records} records` : ""}
                  </p>
                </div>
              ))}
            </div>
          </section>
        </aside>

        <section className="space-y-2">
          {papers.length === 0 ? (
            <div className="panel px-4 py-6 text-center">
              <p className="font-mono text-xs text-edge">No papers match this filter.</p>
            </div>
          ) : null}
          {papers.map((paper) => (
            <PaperRow
              key={paper.id}
              paper={paper}
              selected={detail?.id === paper.id}
              onSelect={select}
            />
          ))}
        </section>

        <aside className="panel h-fit px-4 py-3">
          {detail ? (
            <div className="space-y-3">
              <h2 className="font-mono text-sm leading-snug text-phosphor">{detail.title}</h2>
              <p className="font-mono text-3xs text-edge">
                {detail.authors.join(", ")} · {detail.year ?? "n.d."} ·{" "}
                {detail.venue || detail.source}
              </p>
              {detail.url ? (
                <a
                  href={detail.url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-block font-pixel text-3xs text-cyan hover:text-phosphor"
                >
                  OPEN SOURCE ▸
                </a>
              ) : null}
              <p className="font-mono text-3xs leading-relaxed text-[#a9a3e0]">
                {detail.abstract}
              </p>

              <div className="grid grid-cols-2 gap-2">
                <div className="border-2 border-edge px-2 py-1">
                  <p className="stat-label">Relevance</p>
                  <p className="font-mono text-xs text-phosphor">
                    {detail.relevance.toFixed(4)}
                  </p>
                </div>
                <div className="border-2 border-edge px-2 py-1">
                  <p className="stat-label">Citations</p>
                  <p className="font-mono text-xs text-amber">{detail.citation_count}</p>
                </div>
              </div>

              {detail.references.length > 0 ? (
                <div>
                  <p className="stat-label">References ({detail.references.length})</p>
                  <ul className="mt-1 space-y-1">
                    {detail.references.map((ref) => (
                      <li key={ref.id}>
                        <button
                          type="button"
                          onClick={() => select(ref.id)}
                          className="text-left font-mono text-3xs text-[#8f89c9] hover:text-phosphor"
                        >
                          ▸ {ref.title}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {detail.cited_by.length > 0 ? (
                <div>
                  <p className="stat-label">Cited by ({detail.cited_by.length})</p>
                  <ul className="mt-1 space-y-1">
                    {detail.cited_by.map((ref) => (
                      <li key={ref.id}>
                        <button
                          type="button"
                          onClick={() => select(ref.id)}
                          className="text-left font-mono text-3xs text-[#8f89c9] hover:text-phosphor"
                        >
                          ◂ {ref.title}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : (
            <p className="font-mono text-xs text-edge">
              Select a paper to see its abstract, relevance score and citation neighbourhood.
            </p>
          )}
        </aside>
      </div>
    </main>
  );
}
