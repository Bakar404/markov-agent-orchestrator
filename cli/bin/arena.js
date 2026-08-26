#!/usr/bin/env node
/**
 * arena — is orchestration worth it, measured from the terminal.
 *
 * Defaults follow the measured model rates: an Opus orchestrator, Haiku workers of the same
 * family so the variable is capability tier rather than vendor, and a GPT judge so neither arm
 * is graded by its own family.
 */

import { Api } from "../lib/client.js";
import { bad, banner, c, panel, rule, wrapText } from "../lib/ui.js";
import { run } from "../commands/run.js";

const DEFAULTS = {
  api: "http://localhost:8000",
  experiment: "worth-it",
  seeds: [101, 102, 103],
  maxSteps: 6,
  budget: 5.0,
  treatmentArm: "always_orchestrate",
  orchestratorModel: "claude-opus-5",
  workerModel: "claude-haiku-4.5",
  judgeModel: "gpt-5.4",
  judgeExcerpt: 6000,
  allowTools: false,
  hypotheses: [],
};

function parseArgs(argv) {
  const opts = { ...DEFAULTS };
  const positional = [];

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      positional.push(arg);
      continue;
    }
    const [flag, inline] = arg.slice(2).split("=");
    const next = () => inline ?? argv[++i];

    switch (flag) {
      case "task": opts.task = next(); break;
      case "hypothesis": opts.hypotheses.push(next()); break;
      case "experiment": opts.experiment = next(); break;
      case "seeds": opts.seeds = next().split(",").map((s) => Number(s.trim())); break;
      case "steps": opts.maxSteps = Number(next()); break;
      case "budget": opts.budget = Number(next()); break;
      case "arm": opts.treatmentArm = next(); break;
      case "orchestrator": opts.orchestratorModel = next(); break;
      case "worker": opts.workerModel = next(); break;
      case "judge": opts.judgeModel = next(); break;
      case "api": opts.api = next(); break;
      case "allow-tools": opts.allowTools = true; break;
      case "help": opts.help = true; break;
      default:
        console.error(bad(`unknown flag --${flag}`));
        process.exit(1);
    }
  }
  return { opts, positional };
}

function usage() {
  console.log(banner());
  console.log(
    panel("arena run", [
      c.dim("Runs a control arm and an orchestrated arm on the same seeds, then judges"),
      c.dim("them blind. Control runs first. Every agent gets a fresh session."),
      "",
      `${c.phosphor("--task")} <text>            ${c.dim("required")}`,
      `${c.phosphor("--hypothesis")} <text>      ${c.dim("repeat 2-6 times; they must genuinely compete")}`,
      `${c.phosphor("--experiment")} <name>      ${c.dim(`default ${DEFAULTS.experiment}`)}`,
      `${c.phosphor("--seeds")} 101,102,103      ${c.dim("same seeds on both arms, or nothing pairs")}`,
      `${c.phosphor("--steps")} <n>              ${c.dim(`default ${DEFAULTS.maxSteps}`)}`,
      `${c.phosphor("--budget")} <n>             ${c.dim("matched across arms; it is the cost denominator")}`,
      `${c.phosphor("--orchestrator")} <model>   ${c.dim(`default ${DEFAULTS.orchestratorModel}`)}`,
      `${c.phosphor("--worker")} <model>         ${c.dim(`default ${DEFAULTS.workerModel}`)}`,
      `${c.phosphor("--judge")} <model>          ${c.dim(`default ${DEFAULTS.judgeModel}`)}`,
      `${c.phosphor("--allow-tools")}            ${c.dim("let children run shell; slower, and they may search")}`,
      "",
      c.dim("Five seeds is the floor for significance: the two-standard-error bar"),
      c.dim("is 1/sqrt(n), so below five no win rate can clear it."),
    ]),
  );
  console.log(`\n${rule(c.amber("EXAMPLE"), c.edge)}`);
  console.log(
    c.dim(`  arena run --task "our CI suite takes 45 minutes and nobody runs it locally" \\
    --hypothesis "shard across more runners" \\
    --hypothesis "fast pre-merge subset, full suite post-merge" \\
    --hypothesis "fix test design, replace integration with unit" \\
    --hypothesis "merge queue makes local runs unnecessary" \\
    --experiment ci-debt --seeds 101,102,103,104,105\n`),
  );
}

const [, , command = "help", ...rest] = process.argv;
const { opts } = parseArgs(rest);

if (command === "help" || opts.help || !["run", "compare"].includes(command)) {
  usage();
  process.exit(0);
}

if (command === "compare") {
  const api = new Api(opts.api);
  const cmp = await api.compare(opts.experiment);
  console.log(banner());
  console.log(
    panel(cmp.mode === "live" ? "LIVE AGENTS" : cmp.mode.toUpperCase(), [
      ...wrapText(cmp.verdict, 66).map((l) => c.mist(l)),
      "",
      ...cmp.arms.map(
        (a) =>
          `${(a.arm === "control" ? c.cyan : c.magenta)(a.arm.padEnd(20))} ` +
          c.dim(
            `runs ${a.runs}  esc ${a.escalated}/${a.runs}  ` +
              `${Math.round(a.mean_tokens).toLocaleString()} tok  ${a.mean_cost_usd.toFixed(2)} AIU`,
          ),
      ),
    ]),
  );
  if (cmp.caveats?.length) {
    console.log(`\n${c.amber("Read this before believing the table")}`);
    for (const caveat of cmp.caveats) {
      for (const [i, line] of wrapText(caveat, 66).entries()) {
        console.log(`  ${i === 0 ? c.amber("!") : " "} ${c.dim(line)}`);
      }
    }
  }
  process.exit(0);
}

if (!opts.task) {
  console.error(bad("--task is required"));
  process.exit(1);
}
if (opts.hypotheses.length < 2) {
  console.error(
    bad("at least two --hypothesis values are required, and they must genuinely compete"),
  );
  console.error(
    c.dim("  Four rephrasings of one answer collapse the belief trivially and fake a result."),
  );
  process.exit(1);
}

await run(opts);
