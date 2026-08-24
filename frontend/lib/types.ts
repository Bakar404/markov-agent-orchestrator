/** Mirrors the FastAPI response contracts in `backend/app`. */

export interface AgentSpec {
  id: string;
  label: string;
  role: string;
  description: string;
  color: string;
  accent: string;
  base_cost_usd: number;
  base_latency_ms: number;
  base_tokens: number;
  prior_alpha: number;
  prior_beta: number;
  evidence_strength: number;
  noise_strength: number;
  quality_gain: number;
  verification_gain: number;
  memory_gain: number;
  subtask_resolution: number;
  canvas_position: { x: number; y: number };
}

export interface ActionSpec {
  id: string;
  label: string;
  description: string;
  kind: "agent" | "coalition" | "control";
  agent: string | null;
}

export interface PolicySpec {
  id: string;
  label: string;
  stage: number;
  family: string;
  description: string;
}

export interface TaxonomyCategory {
  name: string;
  description: string;
  color: string;
  keywords: string[];
  discovery_queries: string[];
}

export interface Meta {
  app: string;
  version: string;
  agents: AgentSpec[];
  actions: ActionSpec[];
  policies: PolicySpec[];
  strategies: Strategy[];
  taxonomy: TaxonomyCategory[];
  reward_weights: Record<string, number>;
  state_features: string[];
  defaults: { seed: number; max_steps: number; belief_dim: number };
  research: { network_enabled: boolean; mcp_endpoint_configured: boolean };
}

export interface AgentHistory {
  alpha: number;
  beta: number;
  invocations: number;
  successes: number;
  partials: number;
  failures: number;
  last_step: number;
  cumulative_reward: number;
  cumulative_cost: number;
  cumulative_tokens: number;
  cumulative_information_gain: number;
  success_rate: number;
}

export interface OrchestratorState {
  step: number;
  task_complexity: number;
  belief: number[];
  belief_probabilities: number[];
  entropy: number;
  uncertainty: number;
  confidence: number;
  budget_total_usd: number;
  budget_spent_usd: number;
  budget_remaining_usd: number;
  budget_remaining: number;
  latency_budget_ms: number;
  latency_consumed_ms: number;
  latency_remaining: number;
  tokens_consumed: number;
  quality: number;
  verification_score: number;
  memory_coverage: number;
  total_subtasks: number;
  unresolved_subtasks: number;
  unresolved_ratio: number;
  duplicate_pressure: number;
  mean_agent_success: number;
  agent_history: Record<string, AgentHistory>;
  last_agents: string[];
  terminated: boolean;
  termination_reason: string | null;
  latent_hypothesis: number;
  has_escalated: boolean;
  solo_steps: number;
  stall_steps: number;
  stall: number;
  needs_evidence: number;
  needs_execution: number;
  needs_verification: number;
  features: Record<string, number>;
}

export interface Strategy {
  id: string;
  label: string;
  policy: string;
  summary: string;
  when: string;
  category: string;
  paper_query: string;
  is_control: boolean;
  escalates: "never" | "always" | "heuristic" | "learned";
  policy_options: Record<string, unknown>;
}

export interface PairedStat {
  mean_delta: number;
  stderr: number;
  significant: boolean;
  n: number;
  multiple?: number | null;
}

export interface ArmDelta {
  paired_seeds: number;
  note?: string;
  cost_usd?: PairedStat;
  latency_ms?: PairedStat;
  tokens?: PairedStat;
  steps?: PairedStat;
  quality?: PairedStat | null;
}

export interface ExperimentArm {
  arm: string;
  policy: string;
  runs: number;
  seeds: number[];
  goal_reached: number;
  escalated: number;
  mean_cost_usd: number;
  mean_latency_ms: number;
  mean_tokens: number;
  mean_steps: number;
  mean_quality: number | null;
  judged_runs: number;
  mean_internal_reward: number;
  run_ids: string[];
  vs_control: ArmDelta | null;
}

export interface ExperimentComparison {
  experiment: string;
  tasks: string[];
  control_arm: string | null;
  arms: ExperimentArm[];
  verdict: string;
  caveats: string[];
}

export interface ExperimentSummary {
  experiment: string;
  arms: string[];
  runs: number;
  seeds: number[];
  tasks: string[];
  has_control: boolean;
}

export interface RewardBreakdown {
  quality: number;
  verification: number;
  information_gain: number;
  progress: number;
  cost: number;
  latency: number;
  duplicate: number;
  terminal: number;
  total: number;
  per_agent: Record<string, number>;
}

export interface AgentReport {
  agent_id: string;
  outcome: "success" | "partial" | "failure";
  outcome_probability: number;
  competence_sample: number;
  cost_usd: number;
  latency_ms: number;
  tokens: number;
  evidence_mass: number;
  correct_evidence: boolean;
  summary: string;
  source: "simulated" | "live";
  claimed_hypothesis: number | null;
  response_excerpt: string;
}

export interface RunMessage {
  step: number;
  sender: string;
  receiver: string;
  kind: string;
  content: string;
  weight: number;
}

export interface StepResult {
  step: number;
  action: string;
  action_label: string;
  agents: string[];
  action_probability: number;
  action_distribution: Record<string, number>;
  transition_probability: number;
  outcome: string;
  reward: number;
  cumulative_reward: number;
  reward_breakdown: RewardBreakdown;
  entropy_before: number;
  entropy_after: number;
  information_gain: number;
  confidence: number;
  cost_usd: number;
  latency_ms: number;
  tokens: number;
  prev_state: OrchestratorState;
  state: OrchestratorState;
  reports: AgentReport[];
  messages: RunMessage[];
  diagnostics: Record<string, unknown>;
  done: boolean;
  termination_reason: string | null;
  notes: string;
  wall_clock_ms: number;
  legal_actions: string[];
}

export interface RunPreview {
  legal_actions: string[];
  distribution: Record<string, number>;
  diagnostics: Record<string, unknown>;
  expected_agents: Record<string, string | null>;
  preferred_coalition: string[] | null;
}

export interface RunSummary {
  id: string;
  task: string;
  policy: string;
  status: string;
  seed: number;
  step_count: number;
  cumulative_reward: number;
  total_cost: number;
  total_latency_ms: number;
  total_tokens: number;
  terminated: boolean;
  termination_reason: string | null;
  created_at: string | null;
  updated_at: string | null;
  confidence: number;
  entropy: number;
}

export interface RunDetail extends RunSummary {
  config: Record<string, unknown> & { mode?: "sim" | "live"; hypotheses?: string[] };
  state: OrchestratorState;
  initial_state: OrchestratorState;
  preview: RunPreview;
  agents: AgentSpec[];
}

export interface Trace {
  id: string;
  run_id: string;
  step: number;
  state_id: string;
  prev_state_id: string | null;
  timestamp: string | null;
  action: string;
  agents: string[];
  action_probability: number;
  transition_probability: number;
  action_distribution: Record<string, number>;
  outcome: string;
  confidence: number;
  entropy_before: number;
  entropy_after: number;
  information_gain: number;
  reward: number;
  cumulative_reward: number;
  reward_breakdown: RewardBreakdown;
  latency_ms: number;
  cost_usd: number;
  tokens: number;
  notes: string;
}

export interface MetricsSeriesPoint {
  step: number;
  reward: number;
  cumulative_reward: number;
  entropy_before: number;
  entropy_after: number;
  information_gain: number;
  confidence: number;
  cost_usd: number;
  latency_ms: number;
  tokens: number;
  action: string;
  agents: string[];
}

export interface AgentMetrics {
  agent_id: string;
  label: string;
  color: string;
  reward: number;
  cost: number;
  latency_ms: number;
  tokens: number;
  invocations: number;
  information_gain: number;
  cost_efficiency: number;
}

export interface RunMetrics {
  run_id: string;
  totals: {
    cumulative_reward: number;
    total_cost: number;
    total_latency_ms: number;
    total_tokens: number;
    steps: number;
    cost_efficiency: number;
    reward_per_step: number;
    tokens_per_step: number;
    total_information_gain: number;
    quality: number;
    verification_score: number;
    confidence: number;
    entropy: number;
    memory_coverage: number;
  };
  reward_terms: Record<string, number>;
  per_agent: AgentMetrics[];
  series: MetricsSeriesPoint[];
}

export interface InteractionEdge {
  source: string;
  target: string;
  count: number;
  weight: number;
  mean_weight: number;
  kinds: string[];
}

export interface InteractionGraph {
  edges: InteractionEdge[];
  messages: RunMessage[];
}

export interface Paper {
  id: string;
  external_id: string;
  source: string;
  title: string;
  abstract: string;
  authors: string[];
  year: number | null;
  venue: string;
  url: string;
  pdf_url: string;
  citation_count: number;
  relevance: number;
  notes: string;
  discovered_via: string;
  tags: string[];
}

export interface PaperDetail extends Paper {
  references: Paper[];
  cited_by: Paper[];
}

export interface CitationGraphNode extends Paper {
  in_degree: number;
  out_degree: number;
  hub_score: number;
  authority_score: number;
}

export interface CitationGraph {
  nodes: CitationGraphNode[];
  edges: { id: string; source: string; target: string; relation: string; weight: number }[];
  stats: { nodes: number; edges: number; density: number };
}

export interface ResearchStats {
  papers: number;
  citations: number;
  by_source: { source: string; count: number }[];
  by_year: { year: number; count: number }[];
  tags: { tag: string; category: string; count: number }[];
  taxonomy: TaxonomyCategory[];
}

export interface ProviderHealth {
  id: string;
  label: string;
  kind: string;
  description: string;
  homepage: string;
  available?: boolean;
  configured?: boolean;
  mode?: string;
  records?: number;
  endpoint?: string;
  error?: string;
}

export interface SearchResponse {
  query: string;
  providers: {
    provider: string;
    query: string;
    count: number;
    degraded: boolean;
    error: string;
    latency_ms: number;
  }[];
  degraded: boolean;
  count: number;
  papers: Paper[];
}

export interface CreateRunPayload {
  task: string;
  /** Either a strategy id from the catalog, or a raw policy id. */
  strategy?: string;
  policy?: string;
  seed?: number | null;
  task_complexity?: number;
  budget_usd?: number;
  latency_budget_ms?: number;
  max_steps?: number;
  belief_dim?: number;
  stochasticity?: number;
  confidence_target?: number;
  verification_target?: number;
  min_steps_before_terminate?: number;
  experiment?: string;
  arm?: string;
  mode?: "sim" | "live";
  hypotheses?: string[];
  task_shape?: Record<string, number>;
  policy_profile?: string;
}

export interface CampaignEpisode {
  episode: number;
  seed: number;
  reward: number;
  won: boolean;
  steps: number;
  confidence: number;
  cost: number;
  reason: string;
}

export interface CampaignArm {
  mean_reward: number;
  win_rate: number;
  mean_steps: number;
  mean_confidence: number;
  episodes: CampaignEpisode[];
}

export interface CampaignPolicyResult {
  policy: string;
  label: string;
  stage: number;
  carried: CampaignArm;
  fresh: CampaignArm;
  delta: number;
  stderr: number;
  significant: boolean;
  slope: number;
  blocks: number[];
}

export interface CampaignResponse {
  config: {
    episodes: number;
    seed_base: number;
    max_steps: number;
    budget_usd: number;
    task_complexity: number;
    policies: string[];
  };
  results: CampaignPolicyResult[];
  interpretation: string;
}

export interface CampaignPayload {
  policies?: string[];
  episodes?: number;
  seed_base?: number;
  max_steps?: number;
  budget_usd?: number;
  task_complexity?: number;
}

export type SocketEvent =
  | { type: "snapshot"; run: RunDetail }
  | { type: "step"; step: StepResult; run: RunDetail }
  | { type: "status"; playing: boolean; interval_ms: number; status: string | null }
  | { type: "terminated"; run: RunDetail; reason: string | null }
  | { type: "error"; detail: string };
