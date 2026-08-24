import type {
  CampaignPayload,
  CampaignResponse,
  CitationGraph,
  CreateRunPayload,
  ExperimentComparison,
  ExperimentSummary,
  InteractionGraph,
  Meta,
  Paper,
  PaperDetail,
  ProviderHealth,
  ResearchStats,
  RunDetail,
  RunMetrics,
  RunSummary,
  SearchResponse,
  StepResult,
  Trace,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(
      `Cannot reach the orchestration API at ${API_BASE}. Is the backend running?`,
      0,
    );
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      // Non-JSON error body; keep the status line.
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  meta: () => request<Meta>("/api/meta"),

  listRuns: (limit = 25) => request<RunSummary[]>(`/api/runs?limit=${limit}`),
  getRun: (id: string) => request<RunDetail>(`/api/runs/${id}`),
  createRun: (payload: CreateRunPayload) =>
    request<RunDetail>("/api/runs", { method: "POST", body: JSON.stringify(payload) }),
  step: (id: string, steps = 1) =>
    request<{ run: RunDetail; steps: StepResult[] }>(`/api/runs/${id}/step`, {
      method: "POST",
      body: JSON.stringify({ steps }),
    }),
  reset: (id: string, seed?: number, keepPolicyLearning = false) =>
    request<RunDetail>(`/api/runs/${id}/reset`, {
      method: "POST",
      body: JSON.stringify({ seed: seed ?? null, keep_policy_learning: keepPolicyLearning }),
    }),
  deleteRun: (id: string) => request<void>(`/api/runs/${id}`, { method: "DELETE" }),
  traces: (id: string) => request<Trace[]>(`/api/runs/${id}/traces`),
  metrics: (id: string) => request<RunMetrics>(`/api/runs/${id}/metrics`),
  interactionGraph: (id: string) => request<InteractionGraph>(`/api/runs/${id}/graph`),

  experiments: () => request<ExperimentSummary[]>("/api/experiments"),
  experiment: (name: string) =>
    request<ExperimentComparison>(`/api/experiments/${encodeURIComponent(name)}`),
  scoreRun: (id: string, payload: { score: number; judge?: string; rubric?: string; notes?: string }) =>
    request<{ run_id: string; score: number }>(`/api/runs/${id}/verdict`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  researchStats: () => request<ResearchStats>("/api/research/stats"),
  papers: (params: { tag?: string; search?: string; limit?: number } = {}) => {
    const query = new URLSearchParams();
    if (params.tag) query.set("tag", params.tag);
    if (params.search) query.set("search", params.search);
    query.set("limit", String(params.limit ?? 100));
    return request<Paper[]>(`/api/research/papers?${query.toString()}`);
  },
  paper: (id: string) => request<PaperDetail>(`/api/research/papers/${id}`),
  citationGraph: (limit = 220) => request<CitationGraph>(`/api/research/graph?limit=${limit}`),
  providers: () => request<ProviderHealth[]>("/api/research/providers"),
  search: (query: string, providers?: string[], limit = 10) =>
    request<SearchResponse>("/api/research/search", {
      method: "POST",
      body: JSON.stringify({ query, providers: providers ?? null, limit, persist: true }),
    }),
  hits: (limit = 10) =>
    request<{
      algorithm: string;
      authorities: { key: string; title: string; score: number }[];
      hubs: { key: string; title: string; score: number }[];
    }>(`/api/research/hits?limit=${limit}`),

  campaign: (payload: CampaignPayload) =>
    request<CampaignResponse>("/api/campaign", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};

export function runSocketUrl(runId: string): string {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}/ws/runs/${runId}`;
}
