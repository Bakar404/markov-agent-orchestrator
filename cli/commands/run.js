/**
 * `arena run` — a paired experiment, driven from the terminal.
 *
 * Control runs first across every seed so no solo answer follows a specialist one, each brief
 * gets a fresh session, and a third session judges with the labels stripped and the order
 * shuffled. The driver never contributes content: a driver that reasons about the task is an
 * unmeasured third arm.
 */

import { Api, copilotVersion, invokeChild } from "../lib/client.js";
import { AGENT_COLOR, bad, banner, c, ok, panel, roster, rule, warn, wrapText } from "../lib/ui.js";

const RUBRIC =
  "Diagnosis before prescription; discriminating evidence named; specific enough to act on this " +
  "week; tradeoffs stated; coherent and readable. Tiebreak: what a staff engineer could hand " +
  "their team on Monday.";

function buildPrompt(task, brief) {
  const ranked = (brief.context?.belief_ranked ?? [])
    .map((h) => `  [${h.index}] ${h.label}  (p=${h.probability.toFixed(3)})`)
    .join("\n");

  return `You are one agent inside an experiment. Do the work described, then stop.

TASK
${task}

COMPETING HYPOTHESES
${ranked}

YOUR ROLE: ${brief.agent_id}
${brief.instruction}

RULES
- Produce the work itself. Do not describe what you would do.
- If you cannot do something, say so plainly rather than inventing it.
- No preamble, no sign-off. Output only the work.
- End with a final line exactly: HYPOTHESIS: <index>   (or HYPOTHESIS: none)`;
}

const claimedFrom = (text) => {
  const m = text.match(/^HYPOTHESIS:\s*(\d+|none)\s*$/im);
  if (!m || m[1].toLowerCase() === "none") return null;
  return Number(m[1]);
};

async function driveArm(api, runId, task, opts) {
  let unlocked = false;
  for (;;) {
    const run = await api.getRun(runId);
    if (run.terminated || run.step_count >= opts.maxSteps) break;

    const opened = await api.open(runId);

    // No agents means the policy escalated: nothing was produced, so nothing is reported.
    if (!opened.agents?.length) {
      await api.report(runId, { token: opened.token, reports: [] });
      unlocked = true;
      console.log(
        `    ${c.amber("escalated")}  ${roster([], true)}  ${c.dim("specialist roster unlocked")}`,
      );
      continue;
    }

    const reports = [];
    let spend = { tokens: 0, aiu: 0, premium: 0 };

    for (const brief of opened.briefs) {
      const tint = AGENT_COLOR[brief.agent_id] ?? c.mist;
      process.stdout.write(`    ${tint(brief.agent_id.padEnd(12))} ${c.dim("thinking…")}\r`);

      let result;
      try {
        result = await invokeChild(buildPrompt(task, brief), {
          model: brief.model,
          allowTools: opts.allowTools,
        });
      } catch (err) {
        console.log(`    ${bad(`${brief.agent_id}: ${err.message}`)}`);
        await api.abandon(runId).catch(() => {});
        return;
      }

      const body = result.response.replace(/^HYPOTHESIS:.*$/gim, "").trim();
      const firstLine = body.split("\n").find((l) => l.trim())?.trim() ?? "";
      const report = {
        agent_id: brief.agent_id,
        // An empty child response is a failed step, not one to skip or fill in.
        outcome: body ? "success" : "failure",
        confidence: 0.7,
        response: body || `${brief.agent_id} produced no output.`,
        summary: firstLine.slice(0, 160) || `${brief.agent_id}: no output`,
        tokens: result.tokens,
        latency_ms: result.latencyMs,
        cost_usd: result.aiu,
      };
      const claimed = claimedFrom(result.response);
      if (claimed !== null && claimed < opts.hypotheses.length) report.claimed_hypothesis = claimed;
      reports.push(report);

      spend.tokens += result.tokens;
      spend.aiu += result.aiu;
      spend.premium += result.premium;
      process.stdout.write(" ".repeat(60) + "\r");
    }

    let stepped;
    try {
      stepped = await api.report(runId, { token: opened.token, reports });
    } catch (err) {
      // The API refuses replayed or unmeasured work. That is a real failure of this run, not
      // something to retry until it is accepted.
      console.log(`    ${bad("refused")} ${c.dim(err.message.slice(0, 120))}`);
      await api.abandon(runId).catch(() => {});
      return;
    }

    const s = stepped.step;
    const tone = s.outcome === "success" ? c.lime : s.outcome === "failure" ? c.crimson : c.amber;
    console.log(
      `    ${roster(opened.agents, unlocked)}  ${c.dim(`step ${s.step}`)} ` +
        `${tone(s.outcome.padEnd(7))} ${c.dim(
          `${spend.tokens.toLocaleString()} tok · ${spend.aiu.toFixed(2)} AIU · ` +
            `${spend.premium} premium · ΔH ${s.information_gain >= 0 ? "+" : ""}${s.information_gain.toFixed(3)}`,
        )}`,
    );
    if (s.done) break;
  }
}

async function finalAnswer(api, runId) {
  const messages = await api.messages(runId);
  return messages
    .filter((m) => m.kind?.startsWith("report:"))
    .map((m) => m.content)
    .join("\n\n");
}

async function judge(api, { experiment, task, controlId, treatmentId, seed, model, excerpt }) {
  const [a0, b0] = await Promise.all([finalAnswer(api, controlId), finalAnswer(api, treatmentId)]);
  if (!a0 || !b0) {
    console.log(`  ${warn(`seed ${seed}: one arm produced nothing; not judged`)}`);
    return;
  }

  // Judges favour whichever answer they read first, so the order is shuffled per seed.
  const controlIsA = Math.random() < 0.5;
  const clip = (t) => (t.length > excerpt ? `${t.slice(0, excerpt)}\n[truncated]` : t);
  const A = clip(controlIsA ? a0 : b0);
  const B = clip(controlIsA ? b0 : a0);

  const prompt = `You are judging two answers to the same question. You do not know how either was
produced, and you must not speculate about it.

QUESTION
${task}

RUBRIC
${RUBRIC}

ANSWER A
${A}

ANSWER B
${B}

Decide which better satisfies the rubric. A tie is a legitimate verdict and you should use it
when neither is clearly better - do not invent a preference.

Reply with exactly two lines:
VERDICT: A
REASON: <one sentence>`;

  const out = await invokeChild(prompt, { model });
  const m = out.response.match(/^VERDICT:\s*(A|B|TIE)\s*$/im);
  if (!m) {
    console.log(`  ${bad(`seed ${seed}: judge gave no parseable verdict; not recorded`)}`);
    return;
  }

  const choice = m[1].toUpperCase();
  const winner =
    choice === "TIE" ? "tie" : (choice === "A") === controlIsA ? "a" : "b";
  const reason = out.response.match(/^REASON:\s*(.+)$/im)?.[1]?.trim() ?? "";

  await api.pairwise(experiment, {
    run_a: controlId,
    run_b: treatmentId,
    winner,
    judge: `copilot cli (${out.model || model || "default"}, separate session)`,
    rubric: RUBRIC,
    notes: `seed ${seed}; control shown as ${controlIsA ? "A" : "B"}; ${reason}`,
  });

  const label = winner === "tie" ? c.amber("tie") : winner === "a" ? c.cyan("control") : c.magenta("orchestrated");
  console.log(`  ${c.dim(`seed ${seed}`)}  ${label}  ${c.dim(reason.slice(0, 88))}`);
}

/**
 * Everything that can be known to be wrong before a single token is spent.
 *
 * Creating runs first and discovering the problem on the opening brief leaves abandoned rows in
 * the database and a partial experiment that cannot be compared, so each check happens here.
 */
async function preflight(api, opts) {
  let meta;
  try {
    meta = await api.meta();
  } catch {
    return `Backend unreachable at ${opts.api}. Start it with .\\scripts\\dev.ps1`;
  }

  const catalog = meta.strategies ?? [];
  const chosen = catalog.find((s) => s.id === opts.treatmentArm);
  if (!chosen) {
    const ids = catalog.map((s) => s.id).join(", ");
    return `unknown arm '${opts.treatmentArm}'. Available: ${ids}`;
  }
  if (chosen.is_control) {
    return "the treatment arm cannot be the control; the run already includes control";
  }
  if (chosen.external_driver) {
    // This CLI drives agents through Copilot. Those arms are routed by an Agent Framework
    // graph in Python, which this process has no way to run.
    return (
      `'${chosen.id}' is routed by ${chosen.external_driver}, which this CLI cannot drive.\n` +
      `  Run it with:  C:\\venvs\\arena-maf\\Scripts\\python bridge/run.py --arm ${chosen.id}`
    );
  }

  try {
    opts.copilot = await copilotVersion();
  } catch (err) {
    return `${err.message}. Every agent turn and the judge shell out to it.`;
  }
  return null;
}

export async function run(opts) {
  const api = new Api(opts.api);
  const problem = await preflight(api, opts);
  if (problem) {
    console.error(bad(problem));
    process.exitCode = 1;
    return;
  }

  console.log(banner());
  console.log(
    panel("EXPERIMENT", [
      `${c.dim("name")}      ${c.mist(opts.experiment)}`,
      `${c.dim("task")}      ${c.mist(opts.task.slice(0, 52))}${opts.task.length > 52 ? "…" : ""}`,
      `${c.dim("seeds")}     ${c.mist(opts.seeds.join(", "))}`,
      `${c.dim("steps")}     ${c.mist(String(opts.maxSteps))} ${c.dim("max per arm")}`,
      `${c.dim("budget")}    ${c.mist(opts.budget.toFixed(2))} ${c.dim("per arm, matched")}`,
      "",
      `${c.cyan("control")}       ${c.mist(opts.orchestratorModel || "default")} ${c.dim("solo")}`,
      `${c.magenta("orchestrated")}  ${c.mist(opts.orchestratorModel || "default")} ${c.dim("+ workers on")} ${c.mist(opts.workerModel || "default")}`,
      `${c.violet("judge")}         ${c.mist(opts.judgeModel || "default")} ${c.dim("blind, order shuffled")}`,
      "",
      `${c.dim("driver")}    ${c.mist(opts.copilot ?? "copilot")}`,
    ]),
  );

  const workers = {};
  if (opts.workerModel) {
    for (const id of ["planner", "researcher", "critic", "verifier", "memory", "executor"]) {
      workers[id] = opts.workerModel;
    }
  }
  if (opts.orchestratorModel) workers.generalist = opts.orchestratorModel;

  const base = {
    task: opts.task,
    experiment: opts.experiment,
    mode: "live",
    belief_dim: opts.hypotheses.length,
    hypotheses: opts.hypotheses,
    max_steps: opts.maxSteps,
    budget_usd: opts.budget,
    default_model: opts.orchestratorModel,
    cost_unit: "aiu",
  };

  const runs = {};
  for (const seed of opts.seeds) {
    runs[seed] = {
      control: (await api.createRun({ ...base, strategy: "control", arm: "control", seed, agent_models: {} })).id,
      treatment: (
        await api.createRun({
          ...base,
          strategy: opts.treatmentArm,
          arm: opts.treatmentArm,
          seed,
          agent_models: workers,
        })
      ).id,
    };
  }

  // Control first, across every seed.
  for (const phase of [
    { key: "control", label: "control", tint: c.cyan },
    { key: "treatment", label: opts.treatmentArm, tint: c.magenta },
  ]) {
    console.log(`\n${rule(phase.tint(phase.label.toUpperCase()), c.edge)}`);
    for (const seed of opts.seeds) {
      const id = runs[seed][phase.key];
      console.log(`  ${c.dim(`seed ${seed}`)}  ${c.dim(`localhost:3000/?run=${id}`)}`);
      await driveArm(api, id, opts.task, opts);
    }
  }

  console.log(`\n${rule(c.violet("BLIND JUDGING"), c.edge)}`);
  for (const seed of opts.seeds) {
    await judge(api, {
      experiment: opts.experiment,
      task: opts.task,
      controlId: runs[seed].control,
      treatmentId: runs[seed].treatment,
      seed,
      model: opts.judgeModel,
      excerpt: opts.judgeExcerpt,
    });
  }

  const cmp = await api.compare(opts.experiment);
  console.log(`\n${rule(c.phosphor("VERDICT"), c.edge)}`);
  console.log(panel(cmp.mode === "live" ? "LIVE AGENTS" : cmp.mode.toUpperCase(), [
    ...wrapText(cmp.verdict, 62).map((l) => c.mist(l)),
    "",
    ...cmp.arms.map((a) =>
      `${(a.arm === "control" ? c.cyan : c.magenta)(a.arm.padEnd(20))} ` +
      c.dim(
        `runs ${a.runs}  esc ${a.escalated}/${a.runs}  ` +
          `${Math.round(a.mean_tokens).toLocaleString()} tok  ` +
          `${a.mean_cost_usd.toFixed(2)} ${cmp.cost_unit}`,
      ),
    ),
  ], { color: c.phosphor }));

  if (cmp.caveats?.length) {
    console.log(`\n${c.amber("Read this before believing the table")}`);
    for (const caveat of cmp.caveats) {
      for (const [i, line] of wrapText(caveat, 66).entries()) {
        console.log(`  ${i === 0 ? c.amber("!") : " "} ${c.dim(line)}`);
      }
    }
  }

  console.log(
    `\n${ok(`http://localhost:3000/compare?experiment=${encodeURIComponent(opts.experiment)}`)}\n`,
  );
}
