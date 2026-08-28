/**
 * Talks to the arena backend and to Copilot. Nothing here reasons about a task: the driver's
 * only job is to keep the arms and the judge apart, and to record what each call actually spent.
 */

import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

export class Api {
  constructor(base = "http://localhost:8000") {
    this.base = base.replace(/\/$/, "");
  }

  async #json(path, init) {
    const res = await fetch(`${this.base}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
    if (!res.ok) {
      const detail = await res.text();
      const err = new Error(`${res.status} ${path}: ${detail}`);
      err.status = res.status;
      throw err;
    }
    return res.status === 204 ? null : res.json();
  }

  meta = () => this.#json("/api/meta");
  createRun = (body) => this.#json("/api/runs", { method: "POST", body: JSON.stringify(body) });
  getRun = (id) => this.#json(`/api/runs/${id}`);
  messages = (id) => this.#json(`/api/runs/${id}/messages`);
  open = (id) => this.#json(`/api/runs/${id}/live/open`, { method: "POST" });
  report = (id, body) =>
    this.#json(`/api/runs/${id}/live/report`, { method: "POST", body: JSON.stringify(body) });
  abandon = (id) => this.#json(`/api/runs/${id}/live/abandon`, { method: "POST" });
  compare = (name) => this.#json(`/api/experiments/${encodeURIComponent(name)}`);
  pairwise = (name, body) =>
    this.#json(`/api/experiments/${encodeURIComponent(name)}/pairwise`, {
      method: "POST",
      body: JSON.stringify(body),
    });
}

/**
 * Where `copilot` really lives, resolved once.
 *
 * This matters more than it looks. Spawning with `shell: true` hands the argument list to
 * cmd.exe, which concatenates without escaping: a prompt containing `&` is truncated at that
 * character and the remainder is executed. Prompts here carry task text and, for the judge,
 * whole agent answers, so that is both a command injection and a quieter measurement bug where
 * the judge grades a fragment. Resolving a real executable lets every call run with
 * `shell: false`, where Node passes the array through without a shell parsing it.
 */
let resolved = null;

async function which(name) {
  if (process.platform !== "win32") return null;
  return new Promise((resolve) => {
    const child = spawn("where.exe", [name], { shell: false });
    let out = "";
    child.stdout.on("data", (d) => (out += d));
    child.on("error", () => resolve(null));
    child.on("close", (code) =>
      resolve(code === 0 ? out.trim().split(/\r?\n/)[0] || null : null),
    );
  });
}

async function resolveCopilot() {
  if (resolved) return resolved;
  if (process.platform !== "win32") {
    resolved = { command: "copilot", prefix: [] };
    return resolved;
  }

  const exe = await which("copilot.exe");
  if (exe) {
    resolved = { command: exe, prefix: [] };
    return resolved;
  }
  const ps1 = await which("copilot.ps1");
  if (ps1) {
    resolved = {
      command: "powershell.exe",
      prefix: ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1],
    };
    return resolved;
  }
  throw new Error(
    "could not resolve copilot.exe or copilot.ps1 on PATH. " +
      "Install the GitHub Copilot CLI, or run `copilot --version` to check it works.",
  );
}

function runCopilot(args) {
  return resolveCopilot().then(
    ({ command, prefix }) =>
      new Promise((resolve, reject) => {
        const child = spawn(command, [...prefix, ...args], { shell: false });
        let out = "";
        let err = "";
        child.stdout.on("data", (d) => (out += d));
        child.stderr.on("data", (d) => (err += d));
        child.on("error", reject);
        child.on("close", (code) =>
          code === 0 ? resolve(out) : reject(new Error(err.trim() || `copilot exited ${code}`)),
        );
      }),
  );
}

/**
 * Is the Copilot CLI actually callable? Worth asking before a run rather than during one.
 *
 * Every agent turn and the judge all shell out to `copilot`, so a missing binary fails on the
 * first brief, after runs exist and seeds have been created. Checking up front keeps a
 * misconfigured machine from leaving half-finished experiments in the database.
 */
export async function copilotVersion() {
  const out = await runCopilot(["--version"]);
  return out.trim().split("\n")[0];
}

/**
 * One fresh Copilot session. Fresh is the point: an arm that inherits another arm's context is
 * not an independent sample, and a judge that has seen the work is not blind.
 */
export async function invokeChild(prompt, { model = "", allowTools = false } = {}) {
  const usageFile = join(tmpdir(), `arena-usage-${randomUUID()}.json`);
  const args = ["-p", prompt, "-s", "--no-color", "--usage-output-file", usageFile];
  if (model) args.push("--model", model);
  if (allowTools) args.push("--allow-tool", "shell");

  const started = Date.now();
  const stdout = await runCopilot(args);

  let usage;
  try {
    usage = JSON.parse(await readFile(usageFile, "utf8"));
  } catch {
    throw new Error("Copilot wrote no usage file; the call cannot be reported as measured.");
  } finally {
    await rm(usageFile, { force: true });
  }

  const td = usage.tokenDetails ?? {};
  const tokens =
    (td.input?.tokenCount ?? 0) +
    (td.output?.tokenCount ?? 0) +
    (td.cache_write?.tokenCount ?? 0) +
    (td.cache_read?.tokenCount ?? 0);

  return {
    response: stdout.trim(),
    tokens: Math.max(tokens, 1),
    latencyMs: usage.totalApiDurationMs ?? Date.now() - started,
    aiu: (usage.totalNanoAiu ?? 0) / 1e9,
    premium: usage.totalPremiumRequestCost ?? 0,
    model: usage.currentModel ?? model,
  };
}
