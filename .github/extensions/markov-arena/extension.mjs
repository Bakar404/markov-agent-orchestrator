/**
 * Markov Agent Orchestrator — Copilot CLI extension.
 *
 * NOT CURRENTLY LOADED. Copilot CLI v1.0.80 does not discover this file, despite the path
 * matching the one its own UI reports (".github/extensions/"). It was written against the SDK
 * docs bundled with the GitHub Copilot desktop app (v1.1.6), which is a different version.
 * The working path today is the shell + REST protocol in .github/copilot-instructions.md.
 * Kept because the tool definitions are still the right shape if CLI support lands.
 *
 * Turns the arena into a set of tools so a chat session drives the episode. There is no
 * playback timer here: one tool call is one step, and the model itself is the agent being
 * invoked, which is the whole point of live mode.
 *
 * Protocol, which the tool descriptions also spell out for the agent:
 *
 *   arena_start  → create a live run
 *   arena_step   → the POLICY picks who acts next and returns their brief
 *   ...you then actually do the work described in that brief...
 *   arena_report → hand back what you produced; the episode advances one step
 *
 * Never call arena_report without a preceding arena_step: the backend rejects a report that
 * does not match an open step, because the policy's choice is what is being measured.
 */

import { joinSession } from "@github/copilot-sdk/extension";

const API = (process.env.ARENA_API ?? "http://localhost:8000").replace(/\/+$/, "");
const ARENA_UI = (process.env.ARENA_UI ?? "http://localhost:3000").replace(/\/+$/, "");

/** Remembers the active run so the agent does not have to echo ids on every call. */
const current = { runId: null, token: null, agents: [], hypotheses: [] };

async function api(path, { method = "GET", body } = {}) {
    const response = await fetch(`${API}${path}`, {
        method,
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
    });

    const text = await response.text();
    let payload;
    try {
        payload = text ? JSON.parse(text) : null;
    } catch {
        payload = text;
    }

    if (!response.ok) {
        const detail = payload?.detail ?? payload ?? response.statusText;
        throw new Error(`${method} ${path} → HTTP ${response.status}: ${JSON.stringify(detail)}`);
    }
    return payload;
}

function requireRun(runId) {
    const id = runId ?? current.runId;
    if (!id) throw new Error("No active run. Call arena_start first.");
    return id;
}

function summarizeState(run) {
    const s = run.state ?? {};
    return {
        run_id: run.id,
        step: run.step_count,
        cumulative_reward: round(run.cumulative_reward),
        total_cost_usd: round(run.total_cost, 4),
        total_tokens: run.total_tokens,
        confidence: round(s.confidence ?? 0),
        quality: round(s.quality ?? 0),
        verification_score: round(s.verification_score ?? 0),
        unresolved_subtasks: s.unresolved_subtasks,
        terminated: run.terminated,
        termination_reason: run.termination_reason,
    };
}

const round = (value, places = 3) =>
    typeof value === "number" ? Number(value.toFixed(places)) : value;

const tools = [
    {
        name: "arena_start",
        description:
            "Create a live orchestration run. Call this once at the beginning. After it returns, " +
            "call arena_step to find out which agent the policy wants to invoke first.",
        parameters: {
            type: "object",
            properties: {
                task: {
                    type: "string",
                    description: "The task to orchestrate, in natural language.",
                },
                policy: {
                    type: "string",
                    enum: ["random", "heuristic", "contextual_bandit", "mdp", "markov_game", "marl"],
                    description: "Which policy decides who acts. Defaults to marl.",
                },
                hypotheses: {
                    type: "array",
                    items: { type: "string" },
                    description:
                        "Competing candidate answers, one per belief slot. Supply 4 unless you " +
                        "have reason to differ; omit to name them later on the first report.",
                },
                budget_usd: { type: "number", description: "Spend ceiling for the episode." },
                max_steps: { type: "integer", description: "Hard cap on steps. Defaults to 20." },
            },
            required: ["task"],
        },
        handler: async (args) => {
            const hypotheses = args.hypotheses ?? [];
            const run = await api("/api/runs", {
                method: "POST",
                body: {
                    task: args.task,
                    policy: args.policy ?? "marl",
                    mode: "live",
                    belief_dim: hypotheses.length || 4,
                    hypotheses,
                    budget_usd: args.budget_usd ?? 1.5,
                    max_steps: args.max_steps ?? 20,
                },
            });

            current.runId = run.id;
            current.token = null;
            current.hypotheses = hypotheses;

            return JSON.stringify(
                {
                    ...summarizeState(run),
                    watch: `${ARENA_UI}/?run=${run.id}`,
                    next: "Call arena_step to get the first brief.",
                },
                null,
                2,
            );
        },
    },
    {
        name: "arena_step",
        description:
            "Ask the policy who acts next. Returns a brief describing what that agent must do, " +
            "the current belief over hypotheses, and the remaining budget. This does NOT advance " +
            "the episode. After calling it, actually carry out the work in the brief yourself, " +
            "then call arena_report with what you produced.",
        parameters: {
            type: "object",
            properties: {
                run_id: { type: "string", description: "Defaults to the active run." },
            },
        },
        handler: async (args) => {
            const runId = requireRun(args.run_id);
            const pending = await api(`/api/runs/${runId}/live/open`, { method: "POST" });

            current.token = pending.token;
            current.agents = pending.agents;

            if (pending.agents.length === 0) {
                return JSON.stringify(
                    {
                        action: pending.action,
                        note: "The policy chose to TERMINATE. Call arena_report with an empty reports array to close the episode.",
                        token: pending.token,
                    },
                    null,
                    2,
                );
            }

            return JSON.stringify(
                {
                    action: pending.action,
                    action_probability: round(pending.action_probability),
                    token: pending.token,
                    agents: pending.agents,
                    briefs: pending.briefs.map((brief) => ({
                        agent_id: brief.agent_id,
                        label: brief.label,
                        instruction: brief.instruction,
                        hypotheses: brief.hypotheses,
                        belief: brief.context.belief_ranked,
                        entropy_bits: brief.context.entropy_bits,
                        task: brief.context.task,
                    })),
                    next:
                        "Do this work now, then call arena_report with one entry per agent listed " +
                        "in `agents`. Set claimed_hypothesis to the index you actually argued for, " +
                        "or omit it if your work supported none of them.",
                },
                null,
                2,
            );
        },
    },
    {
        name: "arena_report",
        description:
            "Report what the agent(s) actually produced for the currently open step. Requires a " +
            "preceding arena_step. Supply exactly one entry per agent that arena_step listed. " +
            "Be honest about outcome and confidence — the whole measurement depends on it, and " +
            "reporting success for work you did not verify corrupts the run.",
        parameters: {
            type: "object",
            properties: {
                run_id: { type: "string", description: "Defaults to the active run." },
                token: { type: "string", description: "Defaults to the token from arena_step." },
                hypotheses: {
                    type: "array",
                    items: { type: "string" },
                    description:
                        "Name the belief slots. Accepted only once, and only if they were not " +
                        "already named at start. Must match the belief dimension.",
                },
                reports: {
                    type: "array",
                    description: "One entry per agent from the open step.",
                    items: {
                        type: "object",
                        properties: {
                            agent_id: { type: "string" },
                            outcome: {
                                type: "string",
                                enum: ["success", "partial", "failure"],
                                description:
                                    "success = the brief was fulfilled; partial = progress but " +
                                    "incomplete; failure = could not do it or the result was wrong.",
                            },
                            confidence: {
                                type: "number",
                                description: "0-1. How strongly you back the claim. Be calibrated.",
                            },
                            claimed_hypothesis: {
                                type: "integer",
                                description:
                                    "Index of the hypothesis this work supports. Omit if none.",
                            },
                            response: { type: "string", description: "What the agent produced." },
                            summary: { type: "string", description: "One line for the trace." },
                            tokens: { type: "integer", description: "Measured, if known." },
                            latency_ms: { type: "number", description: "Measured, if known." },
                            cost_usd: { type: "number", description: "Measured, if known." },
                        },
                        required: ["agent_id", "outcome", "confidence"],
                    },
                },
            },
            required: ["reports"],
        },
        handler: async (args) => {
            const runId = requireRun(args.run_id);
            const token = args.token ?? current.token;
            if (!token) throw new Error("No open step. Call arena_step first.");

            const body = { token, reports: args.reports ?? [] };
            if (args.hypotheses?.length) body.hypotheses = args.hypotheses;

            const payload = await api(`/api/runs/${runId}/live/report`, {
                method: "POST",
                body,
            });
            current.token = null;

            const { step, run } = payload;
            return JSON.stringify(
                {
                    step: step.step,
                    action: step.action_label,
                    outcome: step.outcome,
                    reward: round(step.reward),
                    cumulative_reward: round(step.cumulative_reward),
                    information_gain_bits: round(step.information_gain, 4),
                    entropy_after_bits: round(step.entropy_after, 4),
                    confidence: round(step.confidence),
                    cost_usd: round(step.cost_usd, 4),
                    reward_breakdown: step.reward_breakdown,
                    done: step.done,
                    termination_reason: step.termination_reason,
                    state: summarizeState(run),
                    next: step.done
                        ? "Episode complete. Call arena_status for the final trace, or arena_start for a new run."
                        : "Call arena_step for the next brief.",
                },
                null,
                2,
            );
        },
    },
    {
        name: "arena_status",
        description:
            "Read the current run state without changing anything: reward, belief, budget, and " +
            "whether a step is currently open.",
        parameters: {
            type: "object",
            properties: {
                run_id: { type: "string", description: "Defaults to the active run." },
            },
        },
        handler: async (args) => {
            const runId = requireRun(args.run_id);
            const run = await api(`/api/runs/${runId}`);
            return JSON.stringify(
                {
                    ...summarizeState(run),
                    step_open: Boolean(current.token),
                    awaiting_reports_from: current.token ? current.agents : [],
                    watch: `${ARENA_UI}/?run=${runId}`,
                },
                null,
                2,
            );
        },
    },
];

const session = await joinSession({
    tools,
    hooks: {
        onSessionStart: async () => {
            await session.log(`Markov arena tools ready (API ${API})`);
        },
    },
});
