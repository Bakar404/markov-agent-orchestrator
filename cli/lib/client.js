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
 * One fresh Copilot session. Fresh is the point: an arm that inherits another arm's context is
 * not an independent sample, and a judge that has seen the work is not blind.
 */
export async function invokeChild(prompt, { model = "", allowTools = false } = {}) {
  const usageFile = join(tmpdir(), `arena-usage-${randomUUID()}.json`);
  const args = ["-p", prompt, "-s", "--no-color", "--usage-output-file", usageFile];
  if (model) args.push("--model", model);
  if (allowTools) args.push("--allow-tool", "shell");

  const started = Date.now();
  const stdout = await new Promise((resolve, reject) => {
    const child = spawn("copilot", args, { shell: process.platform === "win32" });
    let out = "";
    let err = "";
    child.stdout.on("data", (d) => (out += d));
    child.stderr.on("data", (d) => (err += d));
    child.on("error", reject);
    child.on("close", (code) =>
      code === 0 ? resolve(out) : reject(new Error(err.trim() || `copilot exited ${code}`)),
    );
  });

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
